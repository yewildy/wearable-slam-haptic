#!/usr/bin/env python3
"""第一步的候选审核与离线过滤预览；不启动ROS、不改变现场配置。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 一、验证候选资料，不能把任意JSON或已经启用的配置冒充候选。
def checked_candidate(candidate):
    if candidate.get('schema_version') != 1 or candidate.get('kind') != 'initialization_voxel_candidates':
        raise ValueError('Unsupported candidate schema')
    for field in ('self_mask_enabled', 'self_identity_confirmed', 'mask_applied'):
        if candidate.get(field) is not False:
            raise ValueError('Expected an explicitly disabled unconfirmed candidate')
    frame = candidate.get('frame')
    width = candidate.get('parameters', {}).get('voxel_m')
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError('A coordinate frame is required')
    if type(width) not in (int, float) or not np.isfinite(width) or not .005 <= width <= .1:
        raise ValueError('Invalid voxel width')
    rows = candidate.get('candidate_cells')
    if not isinstance(rows, list) or len(rows) > 100000:
        raise ValueError('Invalid candidate cells')
    keys = []
    for row in rows:
        values = row.get('voxel_index') if isinstance(row, dict) else None
        if not isinstance(values, list) or len(values) != 3 or any(type(x) is not int or abs(x) > 1000 for x in values):
            raise ValueError('Invalid voxel index')
        keys.append(tuple(values))
    if len(keys) != len(set(keys)) or candidate.get('candidate_voxels') != len(keys):
        raise ValueError('Duplicate cells or inconsistent candidate count')
    return frame, float(width), keys


# 二、将逐格人工决定变成默认关闭的审核配置。
# SELF只表示审核者声明；软件无法从这个标签证明它确实是身体。
def make_profile(candidate, decisions, candidate_sha256):
    frame, width, keys = checked_candidate(candidate)
    if decisions.get('schema_version') != 1 or decisions.get('candidate_sha256') != candidate_sha256:
        raise ValueError('Review does not match this candidate file')
    rows = decisions.get('decisions')
    if not isinstance(rows, list) or len(rows) != len(keys):
        raise ValueError('Review must account for every candidate cell')
    lookup = set(keys)
    seen, accepted = set(), []
    for row in rows:
        values = row.get('voxel_index') if isinstance(row, dict) else None
        if not isinstance(values, list) or len(values) != 3 or any(type(x) is not int for x in values):
            raise ValueError('Invalid reviewed voxel index')
        key = tuple(values)
        if key not in lookup or key in seen:
            raise ValueError('Unknown or duplicate reviewed cell')
        seen.add(key)
        decision = row.get('decision')
        if decision not in ('UNSURE', 'SELF', 'EXTERNAL'):
            raise ValueError('Invalid review decision')
        reason = row.get('reason', '')
        if not isinstance(reason, str):
            raise ValueError('Reason must be text')
        if decision == 'SELF':
            if not reason.strip():
                raise ValueError('Each SELF cell needs a review reason')
            accepted.append(list(key))
    basis = decisions.get('review_basis', '')
    if not isinstance(basis, str) or (accepted and not basis.strip()):
        raise ValueError('Accepted cells require a stated review basis')
    return {'schema_version': 1, 'kind': 'reviewed_sensor_voxels',
            'self_mask_enabled': False, 'frame': frame, 'voxel_m': width,
            'self_voxels': sorted(accepted), 'candidate_sha256': candidate_sha256,
            'self_identity_confirmed': bool(accepted),
            'confirmation_basis': 'reviewer_declaration_only' if accepted else 'not_confirmed',
            'review_basis': basis, 'review_decisions': rows,
            'masked_space_occupancy': 'unknown_not_free'}


# 三、只预览哪些有效点会被排除。enabled必须显式True，默认全部有效点保留。
# 使用体素索引匹配，不把若干小格扩为大盒，也不修改输入数组。
def preview(points, frame, profile, enabled=False):
    if type(enabled) is not bool:
        raise ValueError('enabled must be a bool')
    if profile.get('kind') != 'reviewed_sensor_voxels' or profile.get('schema_version') != 1:
        raise ValueError('Unsupported review profile')
    if profile.get('self_mask_enabled') is not False:
        raise ValueError('Preview requires a disabled profile')
    if frame != profile.get('frame'):
        raise ValueError('Input coordinate frame mismatch')
    width = profile.get('voxel_m')
    if type(width) not in (int, float) or not np.isfinite(width) or not .005 <= width <= .1:
        raise ValueError('Invalid voxel width')
    accepted = profile.get('self_voxels')
    if not isinstance(accepted, list):
        raise ValueError('Invalid self voxel list')
    keys = set()
    for key in accepted:
        if not isinstance(key, list) or len(key) != 3 or any(type(x) is not int or abs(x) > 1000 for x in key):
            raise ValueError('Invalid self voxel')
        keys.add(tuple(key))
    if enabled and keys and (profile.get('self_identity_confirmed') is not True or
                            profile.get('confirmation_basis') != 'reviewer_declaration_only' or
                            not profile.get('review_basis', '').strip()):
        raise ValueError('No explicit review declaration')
    xyz = np.asarray(points)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind not in 'fi':
        raise ValueError('Expected numeric Nx3 coordinates')
    finite = np.isfinite(xyz).all(axis=1)
    zero = finite & np.all(xyz == 0, axis=1)
    valid = finite & ~zero
    removed = np.zeros(len(xyz), dtype=bool)
    if enabled and keys:
        # 候选输入只在0.5米内，键范围也已限制。更远坐标无需转整数，以免溢出。
        local = valid & np.all(np.abs(xyz) <= (1001*width), axis=1)
        cells = np.floor(xyz[local].astype(np.float64)/width).astype(np.int64)
        removed[local] = [tuple(cell) in keys for cell in cells]
    return {'self_removed': removed, 'invalid': ~valid,
            'kept': valid & ~removed, 'masked_space_occupancy': 'unknown_not_free'}


# 四、生成可离线打开的审核页面。JSON中的“<”转义，防止字符串结束script标签。
def make_page(candidate_path, npz_path, html_path):
    candidate_path, npz_path, html_path = map(Path, (candidate_path, npz_path, html_path))
    candidate = json.loads(candidate_path.read_text(encoding='utf-8'))
    frame, _, _ = checked_candidate(candidate)
    if candidate.get('input_sha256') != digest(npz_path):
        raise ValueError('Candidate and NPZ source fingerprint differ')
    from initialization_candidates import checked_sample
    with np.load(npz_path, allow_pickle=False) as sample:
        xyz, ids, _ = checked_sample(sample, frame)
        points = xyz[ids < candidate['frames_used']]
    # 背景点确定性抽样仅用于显示，审核体素不抽样；页面上写明只含0.5米近点。
    step = max(1, int(np.ceil(len(points)/6000)))
    payload = {'candidate': candidate, 'candidate_sha256': digest(candidate_path),
               'points': points[::step].astype(float).tolist(), 'display_step': step}
    template = Path(__file__).with_name('self_review_template.html').read_text(encoding='utf-8')
    html = template.replace('__REVIEW_DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    with html_path.open('x', encoding='utf-8') as stream:
        stream.write(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    page = sub.add_parser('page')
    page.add_argument('--candidate', required=True, type=Path)
    page.add_argument('--npz', required=True, type=Path)
    page.add_argument('--html', required=True, type=Path)
    test = sub.add_parser('preview')
    test.add_argument('--candidate', required=True, type=Path)
    test.add_argument('--decisions', required=True, type=Path)
    test.add_argument('--npz', required=True, type=Path)
    test.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'page':
            make_page(args.candidate, args.npz, args.html)
            print('SAVED_HTML='+str(args.html.resolve()))
            return
        candidate = json.loads(args.candidate.read_text(encoding='utf-8'))
        decisions = json.loads(args.decisions.read_text(encoding='utf-8'))
        profile = make_profile(candidate, decisions, digest(args.candidate))
        if digest(args.npz) != candidate.get('input_sha256'):
            raise ValueError('Candidate and NPZ source fingerprint differ')
        from initialization_candidates import checked_sample
        with np.load(args.npz, allow_pickle=False) as sample:
            xyz, _, _ = checked_sample(sample, profile['frame'])
        filtered = preview(xyz, profile['frame'], profile, enabled=True)
        summary = {'points': len(xyz), 'would_remove': int(filtered['self_removed'].sum()),
                   'would_keep_valid': int(filtered['kept'].sum()),
                   'invalid_points': int(filtered['invalid'].sum()),
                   'live_mask_applied': False, 'preview_only': True,
                   'masked_space_occupancy': 'unknown_not_free'}
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'reviewed_profile.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.output/'preview_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, 'ERROR: %s\n' % error)


if __name__ == '__main__':
    main()
