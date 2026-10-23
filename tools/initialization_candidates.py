#!/usr/bin/env python3
"""第一步的离线学习版：统计稳定近点候选，不执行屏蔽，不确认自身身份。

输入是 capture_near_cloud.py 导出的 near_returns.npz。
只依赖 NumPy 和同目录已有的 audit_mask_risk.py，不需要 ROS 或雷达在线。
Windows 和 Jetson 可使用同一套计算逻辑；实时 ROS 接口留到下一小步。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

# 复用已经测过的输入检查，避免另写一套略有差异的数据约定。
from audit_mask_risk import checked_sample


# 一、检查参数。秒和米是物理单位，比例在 0 到 1 之间。
def validate_parameters(seconds, voxel_m, occupancy, min_frames, max_gap):
    for value in (seconds, voxel_m, occupancy, max_gap):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Parameters must be finite numbers')
    if not 1 <= seconds <= 60:
        raise ValueError('seconds must be within 1..60')
    if not .005 <= voxel_m <= .1:
        raise ValueError('voxel_m must be within 0.005..0.1')
    if not .5 <= occupancy <= 1:
        raise ValueError('occupancy must be within 0.5..1')
    if type(min_frames) is not int or min_frames < 5:
        raise ValueError('min_frames must be an integer >= 5')
    if not 0 < max_gap <= 1:
        raise ValueError('max_gap must be within 0..1 seconds')


# 二、纯计算函数。没有文件写入、网络操作或 ROS 发布，方便单独测试。
def analyze(sample, expected_frame, seconds=10.0, voxel_m=.02,
            occupancy=.8, min_frames=30, max_gap=.3):
    validate_parameters(seconds, voxel_m, occupancy, min_frames, max_gap)
    # xyz 是 N 行三列坐标；ids 是每个点属于第几帧；stamps 是每帧时间。
    # 检查坐标有限、非零、距离不超过导出的 0.5m、索引合法、坐标系一致、时间递增。
    xyz, ids, stamps = checked_sample(sample, expected_frame)
    elapsed = stamps - stamps[0]
    if elapsed[-1] + 1e-6 < seconds:
        raise ValueError('Recording does not cover the requested initialization duration')
    # 按时间选前10秒，不把“100帧”当作任何情况下都等于10秒。
    # 小容差仅处理时间戳浮点减法误差，不用于放宽实际采样时长。
    frame_count = int(np.count_nonzero(elapsed <= seconds + 1e-6))
    if frame_count < min_frames:
        raise ValueError('Too few frames in initialization window')
    # 包括跨越窗口结束时刻的那一个间隔，避免窗口末尾长断流被忽略。
    checked_end = min(frame_count + 1, len(stamps))
    largest_gap = float(np.max(np.diff(stamps[:checked_end])))
    if largest_gap > max_gap + 1e-6:
        raise ValueError('Initialization window contains an excessive timestamp gap')

    # 三、把窗口分为五段。持续出现比“某一瞬间点特别多”更符合初始化的目的。
    # 每一段至少要有一帧；一帧没有近点，也仍然留在分母中。
    segments = np.minimum((elapsed[:frame_count] / seconds * 5).astype(int), 4)
    segment_sizes = np.bincount(segments, minlength=5)
    if np.any(segment_sizes == 0):
        raise ValueError('Some initialization time segments contain no frames')
    selected = ids < frame_count
    points, point_frames = xyz[selected], ids[selected]

    # 四、体素就是小立方格子。默认边长2厘米，原点固定在雷达坐标系原点。
    # floor(-0.01 / 0.02)=-1，不能用向零取整，否则负坐标的格子会划错。
    # 同一格子同一帧即使有100个点，也只算“这一帧出现过一次”。
    cells = np.floor(points.astype(np.float64) / voxel_m).astype(np.int64)
    pairs = np.unique(np.column_stack((cells, point_frames)), axis=0)
    if len(pairs):
        keys, inverse = np.unique(pairs[:, :3], axis=0, return_inverse=True)
        hits = np.bincount(inverse, minlength=len(keys))
        segment_hits = np.zeros((len(keys), 5), dtype=int)
        np.add.at(segment_hits, (inverse, segments[pairs[:, 3]]), 1)
    else:
        keys = np.empty((0, 3), dtype=np.int64)
        hits = np.empty(0, dtype=int)
        segment_hits = np.empty((0, 5), dtype=int)

    # 五、选候选。总出现率至少80%，同时五段各至少50%。这是探索参数，非身份置信度。
    fractions = hits / frame_count
    segment_fractions = segment_hits / segment_sizes
    keep = (fractions >= occupancy) & np.all(segment_fractions >= .5, axis=1)
    rows = []
    for index in np.flatnonzero(keep):
        cell = keys[index]
        rows.append({'voxel_index': cell.tolist(),
                     'center_m': ((cell + .5) * voxel_m).tolist(),
                     'frames_seen': int(hits[index]),
                     'frame_fraction': float(fractions[index]),
                     'segment_fractions': segment_fractions[index].tolist()})
    # 六、输出只是审核资料，故意不生成可直接载入C++的self_boxes配置。
    # “静止墙面”也可能被选中，这是该方法的信息限制，不是把阈值调高就能解决的事。
    return {'schema_version': 1, 'kind': 'initialization_voxel_candidates',
            'review_state': 'CANDIDATE_UNCONFIRMED',
            'self_mask_enabled': False, 'self_identity_confirmed': False,
            'mask_applied': False, 'frame': expected_frame,
            'parameters': {'seconds': seconds, 'voxel_m': voxel_m,
                           'occupancy': occupancy, 'min_frames': min_frames,
                           'max_gap_seconds': max_gap, 'segments': 5,
                           'min_segment_fraction': .5},
            'frames_used': frame_count, 'near_points_used': int(len(points)),
            'selected_frame_span_seconds': float(elapsed[frame_count-1]),
            'largest_checked_gap_seconds': largest_gap,
            'segment_frame_counts': segment_sizes.tolist(),
            'occupied_voxels': int(len(keys)), 'candidate_voxels': len(rows),
            'candidate_cells': rows,
            'limitations': [
                'Persistence cannot distinguish wearer, bracket, wall or desk.',
                'Only the exported near sample within 0.5 m is inspected.',
                'No declared raw point-count check is possible from this NPZ alone.',
                'Timestamp consistency is not live freshness or motion verification.',
                'Voxel boundary jitter can split persistent surfaces across cells.',
                'No exclusion, object classification, motor control or safety validation.'
            ]}


# 七、文件接口。先完整检查，再创建新目录；已有目录直接报错，不覆盖原包或报告。
def run(npz_path, output, expected_frame, **parameters):
    source = Path(npz_path)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    # allow_pickle=False 拒绝需要执行Python对象反序列化的输入。
    with np.load(source, allow_pickle=False) as archive:
        result = analyze(archive, expected_frame, **parameters)
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError('Input changed during analysis')
    result['input_sha256'] = before  # 只保存在私有报告，公开摘要不带场景指纹或几何。
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'candidate_review.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'candidate_voxels.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ix', 'iy', 'iz', 'x_m', 'y_m', 'z_m', 'frames_seen', 'frame_fraction'])
        for row in result['candidate_cells']:
            writer.writerow(row['voxel_index'] + row['center_m'] +
                            [row['frames_seen'], row['frame_fraction']])
    return result


# 八、命令行入口。后续实时节点可以调用同一纯函数，不必把所有逻辑写进回调。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frame', required=True, help='Recorded sensor coordinate frame')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--voxel-m', type=float, default=.02)
    parser.add_argument('--occupancy', type=float, default=.8)
    parser.add_argument('--min-frames', type=int, default=30)
    parser.add_argument('--max-gap', type=float, default=.3)
    args = parser.parse_args()
    try:
        result = run(args.npz, args.output, args.frame, seconds=args.seconds,
                     voxel_m=args.voxel_m, occupancy=args.occupancy,
                     min_frames=args.min_frames, max_gap=args.max_gap)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, 'ERROR: %s\n' % error)
    print(json.dumps({key: result[key] for key in
                      ('frames_used', 'candidate_voxels', 'self_mask_enabled', 'review_state')}))
    print('SAVED_DIRECTORY=' + str(args.output.resolve()))


if __name__ == '__main__':
    main()
