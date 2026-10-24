"""只读离线诊断：检查剩余近点的时间证据，不改运行阈值或屏蔽结果。

合并相邻小格的桶命中只用于比较，不能据此把点认定为身体或直接删除。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def history_buckets(history, stamp):
    """复现运行版过去10秒的20个桶；当前帧不投票。"""
    buckets = [set() for _ in range(20)]
    for t, cells in history:
        if t < stamp - 10. - 1e-6:
            continue
        if t >= stamp:
            raise ValueError('History must precede the current frame')
        index = min(19, max(0, int((t - (stamp - 10.)) / .5)))
        buckets[index].update(cells)
    return buckets


def qualifies(hits):
    """现有规则：至少16桶，且每个2秒段至少2桶。"""
    return int(np.count_nonzero(hits)) >= 16 and all(
        np.count_nonzero(hits[i:i+4]) >= 2 for i in range(0, 20, 4))


def evidence(points, buckets):
    """逐点对比单格证据与小容差内合并证据，只返回诊断，不执行过滤。

使用与运行版相同的点到2cm小格表面距离及1cm容差。合并证据允许
不同小格共同投票，因此只是一项候选解释，也可能混合不同外部物体。
"""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    cells = sorted(set().union(*buckets))
    keys = np.asarray(cells, dtype=float).reshape(-1, 3)
    centers = (keys + .5) * .02
    votes = np.array([[cell in bucket for bucket in buckets] for cell in cells],
                     dtype=bool).reshape(-1, 20)
    lookup = {cell: i for i, cell in enumerate(cells)}
    stable = np.array([qualifies(row) for row in votes], dtype=bool)
    rows = []
    for point in points:
        cell = tuple(np.floor(point / .02).astype(int))
        own = votes[lookup[cell]] if cell in lookup else np.zeros(20, dtype=bool)
        delta = np.maximum(np.abs(centers-point) - .01, 0.)
        neighbors = np.linalg.norm(delta, axis=1) <= .01+1e-9
        pooled = np.any(votes[neighbors], axis=0) if neighbors.any() else np.zeros(20, dtype=bool)
        rows.append(dict(own_buckets=int(own.sum()),
                         max_neighbor_buckets=int(votes[neighbors].sum(axis=1).max()) if neighbors.any() else 0,
                         pooled_buckets=int(pooled.sum()),
                         pooled_segments=[int(pooled[i:i+4].sum()) for i in range(0, 20, 4)],
                         matches_existing_stable=bool(np.any(stable & neighbors)),
                         pooled_qualifies=qualifies(pooled)))
    return rows


def group_indices(points, radius=.08):
    """只对少量剩余点列完整连通组，不截断边；不是线上算法。"""
    if len(points) > 4000:
        raise ValueError('Diagnostic group enumeration is bounded at 4000 points')
    unseen = np.ones(len(points), dtype=bool)
    result = []
    for seed in range(len(points)):
        if not unseen[seed]:
            continue
        row = [seed]
        unseen[seed] = False
        for index in row:
            adjacent = np.flatnonzero(unseen & (np.linalg.norm(points-points[index], axis=1) <= radius))
            unseen[adjacent] = False
            row.extend(adjacent.tolist())
        result.append(row)
    return result


def synthetic_boundary(backend):
    """2mm边界抖动与完全固定的成团点对照，均不是真实场景。"""
    from motion_pipeline import MotionEngine
    outputs = {}
    for name, jitter in (('fixed', False), ('boundary_jitter_2mm', True)):
        engine = MotionEngine(backend=backend)
        for i in range(81):
            stamp = 1. + i * .125
            x = .119 if not jitter or (i // 4) % 2 == 0 else .121
            near = np.array([[x,.001,.001], [x,.003,.001], [x,.001,.003], [x,.003,.003]])
            far = np.array([[.8,0,0], [.8,.01,0], [.8,0,.01], [.8,.01,.01]])
            history = list(engine.rolling.frames)
            report = engine.process(np.vstack([near, far]), 'livox_frame', stamp)
        stats = evidence(near, history_buckets(history, stamp))
        outputs[name] = dict(state=report['state'], distance_m=report['distance_m'],
                             removed=report['rolling_removed_points'],
                             nearest_evidence=stats[0])
    assert outputs['fixed']['distance_m'] == .8
    assert outputs['boundary_jitter_2mm']['state'] == 'NEAR_FIELD_ALERT'
    assert outputs['boundary_jitter_2mm']['nearest_evidence']['pooled_qualifies']
    return outputs


def audit(code, sample, output, backend='scipy'):
    sys.path.insert(0, str(code))
    from motion_pipeline import MotionEngine
    digest = hashlib.sha256(sample.read_bytes()).hexdigest()
    data = np.load(sample, allow_pickle=False)
    stamps, offsets, xyz = data['stamps'], data['offsets'], data['xyz']
    assert len(offsets) == len(stamps)+1 and offsets[0] == 0 and offsets[-1] == len(xyz)
    assert np.all(np.diff(stamps) > 0) and np.all(np.diff(stamps) <= .5)
    assert np.isfinite(xyz).all() and np.linalg.norm(xyz, axis=1).max() <= .5+1e-8
    engine = MotionEngine(backend=backend)
    frames, point_rows, sizes, nearest_rows = [], [], [], []
    checked_raw = 0
    for index, original in enumerate(stamps):
        stamp = float(original-stamps[0]+1.)
        points = xyz[offsets[index]:offsets[index+1]]
        history = list(engine.rolling.frames)
        report = engine.process(points, 'livox_frame', stamp)
        if not report.get('rolling_ready'):
            continue
        buckets = history_buckets(history, stamp)
        # 对原始整帧逐点复算实际删除掩码，证明诊断没有误用时间窗或容差。
        raw_evidence = evidence(points, buckets)
        predicted = np.array([r['matches_existing_stable'] for r in raw_evidence])
        np.testing.assert_array_equal(predicted, engine.rolling.removed)
        checked_raw += len(points)
        remaining = engine.base.filtered_points
        valid = np.isfinite(remaining).all(axis=1) & (np.linalg.norm(remaining, axis=1) > 0)
        remaining = remaining[valid]
        rows = evidence(remaining, buckets)
        groups = group_indices(remaining)
        assert all(len(g) >= 4 for g in groups)
        assert not any(r['matches_existing_stable'] for r in rows)
        point_rows.extend(rows)
        sizes.extend(len(g) for g in groups)
        if len(remaining):
            nearest_rows.append(rows[int(np.argmin(np.linalg.norm(remaining, axis=1)))])
        frames.append(dict(frame_index=index, state=report['state'], valid=report['valid'],
                           distance_m=report['distance_m'], group_sizes=[len(g) for g in groups],
                           points=rows, protected=report['rolling_protected_track_points']))
    controls = synthetic_boundary(backend)
    summary = dict(sample_frames=len(stamps), mature_frames=len(frames),
                   saved_crop_only=True, real_pose_available=False,
                   mature_states=dict(Counter(r['state'] for r in frames)),
                   frames_with_remaining_points=sum(bool(r['points']) for r in frames),
                   remaining_point_observations=len(point_rows),
                   remaining_group_observations=len(sizes),
                   group_size_histogram=dict(Counter(sizes)),
                   own_bucket_histogram=dict(Counter(r['own_buckets'] for r in point_rows)),
                   pooled_bucket_histogram=dict(Counter(r['pooled_buckets'] for r in point_rows)),
                   pooled_qualifying_points=sum(r['pooled_qualifies'] for r in point_rows),
                   pooled_qualifying_nearest_frames=sum(r['pooled_qualifies'] for r in nearest_rows),
                   existing_stable_matching_residual_points=0,
                   raw_mask_matches_checked=checked_raw,
                   track_protection_restorations=sum(r['protected'] for r in frames),
                   synthetic_controls=controls, runtime_changed=False,
                   pooled_rule_applied=False, identity_and_safety_validated=False)
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == digest
    output.mkdir(exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    (output/'per_frame_private.json').write_text(json.dumps(frames, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('reference','scipy'), default='scipy')
    args = parser.parse_args()
    audit(args.code, args.sample, args.output, args.backend)
