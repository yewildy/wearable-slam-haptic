"""Offline geometry audit, not a proximity detector or self calibration."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def checked_boxes(boxes):
    if not isinstance(boxes, list) or not boxes:
        raise ValueError('At least one box is required')
    result = []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError('Each box must have min and max bounds')
        for key in ('min', 'max'):
            values = box.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError('Each bound must contain three numbers')
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000 for v in values):
                raise ValueError('Invalid bound')
        lo, hi = (np.asarray(box[key], dtype=np.float32) for key in ('min', 'max'))
        if not np.all(lo < hi):
            raise ValueError('Bounds must remain ordered after float32 conversion')
        result.append((lo, hi))
    return result


def removed_by_boxes(xyz, boxes):
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 coordinates')
    removed = np.zeros(len(points), dtype=bool)
    # PCL CropBox without rotation or translation includes all six faces.
    for lo, hi in checked_boxes(boxes):
        removed |= np.all((points >= lo) & (points <= hi), axis=1)
    return removed


def checked_sample(sample, expected_frame):
    required = {'xyz', 'frame_id', 'frame_stamps', 'frame_names'}
    if not required.issubset(sample):
        raise ValueError('Use a current capture export with per-frame frame_names')
    xyz, ids = np.asarray(sample['xyz']), np.asarray(sample['frame_id'])
    stamps, names = np.asarray(sample['frame_stamps']), np.asarray(sample['frame_names'])
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind not in 'fi':
        raise ValueError('Expected numeric Nx3 coordinates')
    if not len(xyz) or not np.isfinite(xyz).all() or np.any(np.all(xyz == 0, axis=1)):
        raise ValueError('Near sample must contain finite nonzero points')
    if np.any(np.linalg.norm(xyz.astype(float), axis=1) > .500001):
        raise ValueError('Expected only the exported 0 to 0.5 m near sample')
    if ids.shape != (len(xyz),) or ids.dtype.kind not in 'iu':
        raise ValueError('Invalid point-to-frame indices')
    if stamps.ndim != 1 or len(stamps) < 2 or stamps.dtype.kind not in 'fi':
        raise ValueError('At least two timestamped frames required')
    if not np.isfinite(stamps).all() or np.any(stamps <= 0) or np.any(np.diff(stamps) <= 0):
        raise ValueError('Frame timestamps must be finite positive and increasing')
    if np.any(ids < 0) or np.any(ids >= len(stamps)):
        raise ValueError('Point-to-frame index out of bounds')
    if names.shape != stamps.shape or names.dtype.kind not in 'US':
        raise ValueError('Invalid per-frame names')
    if not isinstance(expected_frame, str) or not expected_frame.strip() or np.any(names != expected_frame):
        raise ValueError('Geometry and all recorded frames must match')
    return xyz.astype(np.float32), ids.astype(np.int64), stamps.astype(float)


def partition_counts(ids, removed, first, end):
    selected = (ids >= first) & (ids < end)
    counts = np.bincount(ids[selected], minlength=end)[first:end]
    removed_counts = np.bincount(ids[selected & removed], minlength=end)[first:end]
    total, deleted = int(counts.sum()), int(removed_counts.sum())
    return {'frame_count': end-first, 'near_point_count': total,
            'removed_near_points': deleted, 'remaining_near_points': total-deleted,
            'removed_fraction': deleted/total if total else None,
            'frames_without_near_points': int(np.count_nonzero(counts == 0)),
            'frames_with_all_near_points_removed': int(np.count_nonzero((counts > 0) & (counts == removed_counts))),
            'removed_points_per_frame': {'min': int(removed_counts.min()),
                                         'mean': float(removed_counts.mean()),
                                         'max': int(removed_counts.max())}}


def synthetic_risks():
    box = {'min': [-.4, -.1, -.1], 'max': [-.2, .1, .1]}
    pole = np.array([[-.3, 0, z] for z in np.linspace(-.08, .08, 9)], dtype=np.float32)
    straddling = np.array([[-.3, 0, z] for z in np.linspace(-.12, .12, 9)], dtype=np.float32)
    lo, hi = checked_boxes([box])[0]
    faces, outside = [], []
    for axis in range(3):
        for bound, toward in ((lo, -np.inf), (hi, np.inf)):
            p = (lo+hi)/2
            p[axis] = bound[axis]
            faces.append(p.copy())
            p[axis] = np.nextafter(p[axis], np.float32(toward))
            outside.append(p.copy())
    fixtures = [('external_thin_pole_inside_box', pole),
                ('external_thin_pole_crossing_box', straddling),
                ('six_exact_faces', np.array(faces)),
                ('six_one_float32_step_outside', np.array(outside))]
    return [{'case': name, 'input_points': len(points),
             'removed_points': int(removed_by_boxes(points, [box]).sum()),
             'remaining_points': int((~removed_by_boxes(points, [box])).sum())}
            for name, points in fixtures]


def audit(sample, profile, margin=.02):
    if not isinstance(profile, dict) or profile.get('self_identity_confirmed') is not False:
        raise ValueError('This audit requires an explicitly unconfirmed candidate')
    if profile.get('self_mask_enabled') is not False:
        raise ValueError('Input candidate must remain disabled')
    checked_boxes(profile.get('self_boxes'))
    if type(margin) not in (float, int) or not math.isfinite(margin) or not 0 <= margin <= .1:
        raise ValueError('Margin must be between 0 and 0.1 m')
    xyz, ids, stamps = checked_sample(sample, profile.get('self_mask_frame'))
    split = len(stamps)//2
    fit = xyz[ids < split]
    if not len(fit) or not np.any(ids >= split):
        raise ValueError('Both temporal partitions require near points')
    fitted = [{'min': (fit.min(axis=0)-margin).astype(float).tolist(),
               'max': (fit.max(axis=0)+margin).astype(float).tolist()}]
    supplied_removed = removed_by_boxes(xyz, profile['self_boxes'])
    fit_removed = removed_by_boxes(xyz, fitted)
    return {'schema_version': 1, 'analysis_kind': 'geometry_only_near_sample',
            'mask_provenance': 'candidate_unconfirmed_disabled_input',
            'self_identity_confirmed': False, 'live_defaults_changed': False,
            'sample_frames': len(stamps), 'sample_near_points': len(xyz),
            'header_duration_seconds': float(stamps[-1]-stamps[0]),
            'temporal_split': {'fit_frame_indices': [0, split],
                               'heldout_frame_indices': [split, len(stamps)],
                               'index_convention': 'half-open ranges',
                               'independent_scene': False,
                               'note': 'Later frames from the same static recording, not independent scene validation'},
            'supplied_all_data_candidate': partition_counts(ids, supplied_removed, 0, len(stamps)),
            'prefix_fitted_box': fitted, 'fit_margin_m': margin,
            'prefix_fit': partition_counts(ids, fit_removed, 0, split),
            'same_scene_heldout': partition_counts(ids, fit_removed, split, len(stamps)),
            'synthetic_geometry_counterexamples': synthetic_risks(),
            'limitations': ['No labels distinguish self from external objects',
                            'Removal fraction is not accuracy or false-positive reduction',
                            'Only near returns exported within 0.5 m; no full detector, PCL support or level replay',
                            'An external object inside a box is removed regardless of identity',
                            'Masked or occluded space has unknown occupancy, not free-space evidence',
                            'Sparse remnants can later fail the support filter; this audit does not measure that loss']}


def public_summary(report):
    # Scene geometry and raw samples are intentionally absent from the public result.
    return {key: value for key, value in report.items() if key != 'prefix_fitted_box'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', type=Path, help='Current near_returns.npz including frame_names')
    parser.add_argument('candidate', type=Path, help='JSON candidate export; arbitrary YAML is not parsed')
    parser.add_argument('output', type=Path, help='New private report directory; must not exist')
    args = parser.parse_args()
    with np.load(args.sample, allow_pickle=False) as data:
        report = audit({key: data[key] for key in data.files}, json.loads(args.candidate.read_text(encoding='utf-8')))
    report['input_npz_sha256'] = hashlib.sha256(args.sample.read_bytes()).hexdigest()
    report['candidate_sha256'] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'mask_risk_private.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    public = public_summary(report)
    for key in ('input_npz_sha256', 'candidate_sha256'):
        public.pop(key, None)
    (args.output/'mask_risk_summary.json').write_text(json.dumps(public, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(public, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
