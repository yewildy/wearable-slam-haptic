import copy
from types import SimpleNamespace
import unittest

import numpy as np

from study_support_radius import RADII, check_near_export, public_summary, select_indices, support_trend

KEYS = [('fixture', 1), ('fixture', 2)]


def phases():
    return [{'radius_m': radius, 'rows': [{'key': key, 'state': 'STRONG',
             'values': {'supported_points': str(i+3), 'distance_m': str(.5-i*.1)}} for key in KEYS]}
            for i, radius in enumerate(RADII)]


class RadiusStudyTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_bounded(self):
        self.assertEqual(select_indices(141), list(range(0, 141, 10)))
        self.assertEqual(select_indices(2), [0, 1])
        self.assertEqual(select_indices(1), [0])

    def test_invalid_sampling_rejected(self):
        for total in [0, -1, True, 2.5]:
            with self.assertRaises(ValueError):
                select_indices(total)

    def test_monotonic_count_and_distance(self):
        result = support_trend(phases(), KEYS)
        self.assertTrue(result['nondecreasing_supported_counts_all_frames'])
        self.assertTrue(result['nonincreasing_nearest_supported_distance'])
        self.assertFalse(result['accuracy_improvement_established'])

    def test_count_decrease_is_reported(self):
        rows = phases()
        rows[2]['rows'][0]['values']['supported_points'] = '1'
        self.assertFalse(support_trend(rows, KEYS)['nondecreasing_supported_counts_all_frames'])

    def test_missing_and_duplicate_frames_rejected(self):
        for bad in ['missing', 'duplicate']:
            rows = phases()
            if bad == 'missing':
                rows[0]['rows'].pop()
            else:
                rows[0]['rows'].append(copy.deepcopy(rows[0]['rows'][0]))
            with self.assertRaises(ValueError):
                support_trend(rows, KEYS)

    def test_invalid_count_rejected(self):
        for value in ['nan', '-1', '1.5', None]:
            rows = phases()
            rows[0]['rows'][0]['values']['supported_points'] = value
            with self.assertRaises(ValueError):
                support_trend(rows, KEYS)

    def test_unknown_distances_not_imputed(self):
        rows = phases()
        for phase in rows:
            for row in phase['rows']:
                row['values']['distance_m'] = 'nan'
        result = support_trend(rows, KEYS)
        self.assertEqual(result['finite_distance_triplets'], 0)
        self.assertIsNone(result['nonincreasing_nearest_supported_distance'])

    def test_near_export_matches_only_finite_nonzero_near_returns(self):
        points = [(.3, 0, 0), (0, 0, 0), (float('nan'), 0, 0), (.8, 0, 0)]
        message = SimpleNamespace(points=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in points],
            header=SimpleNamespace(frame_id='fixture', stamp=SimpleNamespace(to_sec=lambda: 1.)))
        export = {'xyz': np.array([[.3, 0, 0]], dtype=np.float32), 'frame_id': np.array([0]),
                  'frame_stamps': np.array([1.]), 'frame_names': np.array(['fixture'])}
        self.assertEqual(check_near_export(message, 0, export), 1)
        export['xyz'][0, 0] = .31
        with self.assertRaises(ValueError):
            check_near_export(message, 0, export)

    def test_public_report_omits_raw_rows_and_configuration(self):
        names = ['scope', 'sample_selection', 'parameters_fixed', 'radii_m',
                 'near_export_selected_frames_exact_match', 'saved_sample_results', 'support_trend',
                 'synthetic_results', 'temporary_master_stopped', 'live_defaults_changed', 'limitations']
        report = {key: {} for key in names}
        report['private_phase_rows'] = ['not public']
        result = public_summary(report)
        self.assertNotIn('private_phase_rows', result)
        result['sample_selection']['changed'] = True
        self.assertEqual(report['sample_selection'], {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
