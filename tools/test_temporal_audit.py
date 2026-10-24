import unittest
import importlib.util
import numpy as np
from audit_temporal_support import evidence, group_indices, history_buckets, qualifies, synthetic_boundary


class EvidenceTests(unittest.TestCase):
    def test_fixed_and_new_cells(self):
        rows = evidence([[.119,0,0], [.3,0,0]], [{(5,0,0)} for _ in range(20)])
        self.assertTrue(rows[0]['matches_existing_stable'])
        self.assertFalse(rows[1]['pooled_qualifies'])

    def test_neighbor_votes_do_not_count_twice(self):
        buckets = [{(5,0,0),(6,0,0)} if i < 10 else set() for i in range(20)]
        row = evidence([[.12,0,0]], buckets)[0]
        self.assertEqual(row['pooled_buckets'], 10)
        self.assertFalse(row['pooled_qualifies'])

    def test_boundary_split_is_diagnostic_only(self):
        buckets = [{(5 if i%2 else 6,0,0)} for i in range(20)]
        row = evidence([[.119,0,0]], buckets)[0]
        self.assertEqual(row['own_buckets'], 10)
        self.assertEqual(row['pooled_buckets'], 20)
        self.assertTrue(row['pooled_qualifies'])
        self.assertFalse(row['matches_existing_stable'])

    def test_each_segment_required(self):
        self.assertFalse(qualifies([True]*16+[False]*4))

    def test_window_and_current_frame(self):
        buckets = history_buckets([(0.,{(1,0,0)}),(1.,{(2,0,0)}),(10.9,{(3,0,0)})],11.)
        self.assertNotIn((1,0,0),set().union(*buckets))
        self.assertIn((2,0,0),buckets[0])
        with self.assertRaises(ValueError): history_buckets([(11.,{(2,0,0)})],11.)

    def test_empty_and_group_chain(self):
        self.assertEqual(evidence([[.1,0,0]], [set() for _ in range(20)])[0]['pooled_buckets'],0)
        self.assertEqual(group_indices(np.empty((0,3))),[])
        groups = group_indices(np.array([[0,0,0],[.07,0,0],[.14,0,0],[.21,0,0],[1,0,0]]))
        self.assertEqual([len(g) for g in groups],[4,1])

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'Pipeline tracker requires SciPy')
    def test_synthetic_pipeline_reference(self):
        result = synthetic_boundary('reference')
        self.assertEqual(result['boundary_jitter_2mm']['removed'],0)


if __name__ == '__main__': unittest.main()
