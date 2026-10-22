import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compare_saved_bag import METRICS, compare_rows, describe, finite_number, public_summary, read_expected, summarize


KEYS = [('fixture', 100), ('fixture', 200)]


def row(index, distance='.3', state='STRONG', **values):
    data = {name: '0' for name in METRICS}
    data.update(distance_m=distance, processing_ms='2', level='3', near_field_uncertain='false')
    data.update(values)
    return {'key': KEYS[index], 'state': state, 'values': data}


class ComparisonTests(unittest.TestCase):
    def test_complete_unique_headers(self):
        report = summarize([row(0), row(1)], KEYS)
        self.assertTrue(report['complete_unique_headers'])
        self.assertEqual(report['metrics']['distance_m']['count'], 2)

    def test_missing_frame_is_incomplete(self):
        report = summarize([row(0)], KEYS)
        self.assertEqual(report['missing_frames'], 1)
        self.assertFalse(report['complete_unique_headers'])

    def test_duplicate_not_double_counted(self):
        report = summarize([row(0), row(0), row(1)], KEYS)
        self.assertEqual(report['frames_matched'], 2)
        self.assertEqual(report['duplicate_outputs'], 1)
        self.assertFalse(report['complete_unique_headers'])

    def test_unexpected_frame_not_in_distance_statistics(self):
        extra = row(0, '9')
        extra['key'] = ('wrong', 100)
        report = summarize([row(0), row(1), extra], KEYS)
        self.assertEqual(report['unexpected_outputs'], 1)
        self.assertEqual(report['metrics']['distance_m']['max'], .3)

    def test_nan_unknown_not_zero_distance(self):
        report = summarize([row(0, 'nan', 'NO_SUPPORTED_POINTS', level='UNKNOWN'), row(1)], KEYS)
        self.assertEqual(report['unknown_level_frames'], 1)
        self.assertEqual(report['metrics']['distance_m']['count'], 1)
        self.assertEqual(report['metrics']['distance_m']['min'], .3)

    def test_near_uncertainty_distinct_from_unknown_level(self):
        report = summarize([row(0, near_field_uncertain='true'), row(1)], KEYS)
        self.assertEqual(report['near_field_uncertain_frames'], 1)
        self.assertEqual(report['unknown_level_frames'], 0)

    def test_missing_timing_not_imputed(self):
        first = row(0)
        del first['values']['processing_ms']
        report = summarize([first, row(1)], KEYS)
        self.assertEqual(report['metrics']['processing_ms']['count'], 1)

    def test_missing_level_is_visible_not_reported_as_known(self):
        first = row(0)
        del first['values']['level']
        self.assertEqual(summarize([first], KEYS)['invalid_or_missing_level_frames'], 1)

    def test_frame_mismatch_visible_even_if_unexpected(self):
        bad = row(0, 'nan', 'SELF_MASK_FRAME_MISMATCH', level='UNKNOWN')
        bad['key'] = ('wrong', 100)
        report = summarize([bad], KEYS)
        self.assertEqual(report['frame_mismatch_outputs'], 1)
        self.assertEqual(report['missing_frames'], 2)

    def test_pair_by_headers_not_order(self):
        report = compare_rows([row(0, '.2'), row(1, '.3')], [row(1, '.9'), row(0, '.8')], KEYS)
        self.assertEqual(report['paired_headers'], 2)
        self.assertAlmostEqual(report['candidate_minus_baseline_distance_m']['mean'], .6)

    def test_only_finite_shared_distances_compared(self):
        report = compare_rows([row(0), row(1)], [row(0, 'nan')], KEYS)
        self.assertEqual(report['paired_headers'], 1)
        self.assertEqual(report['finite_distance_pairs'], 0)
        self.assertIsNone(report['candidate_minus_baseline_distance_m']['mean'])

    def test_nonunique_expected_headers_rejected(self):
        with self.assertRaises(ValueError):
            summarize([], [KEYS[0], KEYS[0]])

    def test_numeric_conversion_and_quantiles(self):
        for value in [None, 'nan', 'inf', 'bad', True]:
            self.assertIsNone(finite_number(value))
        self.assertEqual(describe([1., 3.])['p95'], 2.9)

    def test_public_summary_is_allowlisted_copy(self):
        names = ['schema_version', 'scope', 'frames_expected', 'header_duration_seconds', 'header_gap_seconds',
                 'replay_rate', 'parameters', 'mask_identity', 'validation_scope', 'phases', 'paired',
                 'temporary_master_stopped', 'limitations']
        source = {name: {} for name in names}
        source.update(private_parameters={'geometry': 'not public'}, private_frame_rows=['private'])
        result = public_summary(source)
        self.assertNotIn('private_parameters', result)
        self.assertNotIn('private_frame_rows', result)
        result['phases']['changed'] = True
        self.assertEqual(source['phases'], {})

    def test_full_bag_later_frame_change_rejected(self):
        with self.mock_bag([('fixture', 100), ('wrong', 200)]):
            with self.assertRaisesRegex(ValueError, 'Invalid frame'):
                read_expected('unused', 'fixture')

    def test_full_bag_duplicate_header_rejected(self):
        with self.mock_bag([('fixture', 100), ('fixture', 100)]):
            with self.assertRaisesRegex(ValueError, 'strictly increasing'):
                read_expected('unused', 'fixture')

    @staticmethod
    def mock_bag(headers):
        class Bag:
            def __init__(self, *args):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read_messages(self, **kwargs):
                for frame, ns in headers:
                    stamp = SimpleNamespace(to_nsec=lambda value=ns: value)
                    yield '/livox/lidar', SimpleNamespace(header=SimpleNamespace(frame_id=frame, stamp=stamp)), None
        return patch.dict(sys.modules, {'rosbag': SimpleNamespace(Bag=Bag)})


if __name__ == '__main__':
    unittest.main(verbosity=2)
