import json
import unittest
from strength_view_model import ViewModel, checked_report


def report(**kwargs):
    r=dict(valid=True,state='DISTANCE_CANDIDATE',strength=.5,distance_m=1.,self_mask_enabled=False)
    r.update(kwargs)
    return json.dumps(r)


class ModelTests(unittest.TestCase):
    def test_waiting(self):
        self.assertEqual(ViewModel().snapshot(1)['state'],'WAITING_FOR_REPORT')

    def test_valid_values(self):
        for s in (0.,.25,.5,.75,1.):
            m=ViewModel();m.receive(report(strength=s),1.)
            self.assertEqual(m.snapshot(1.5)['strength'],s)

    def test_stale_clears_strength(self):
        m=ViewModel();m.receive(report(),1.)
        r=m.snapshot(1.76)
        self.assertEqual(r['state'],'DISPLAY_STALE');self.assertIsNone(r['strength'])
        self.assertIsNone(r['candidate_distance_m'])

    def test_exact_timeout(self):
        m=ViewModel();m.receive(report(),1.)
        self.assertTrue(m.snapshot(1.75)['valid'])

    def test_recovery(self):
        m=ViewModel();m.receive(report(),1.);m.snapshot(2.)
        m.receive(report(strength=1.),3.)
        self.assertEqual(m.snapshot(3.)['strength'],1.)

    def test_unknown_is_not_zero(self):
        m=ViewModel();m.receive(report(valid=False,state='NEAR_FIELD_UNCERTAIN',strength=None,distance_m=None,candidate_distance_m=.22),1.)
        r=m.snapshot(1.)
        self.assertFalse(r['valid']);self.assertIsNone(r['strength']);self.assertEqual(r['candidate_distance_m'],.22)

    def test_invalid_replaces_old_good(self):
        m=ViewModel();m.receive(report(),1.);m.receive('not json',1.1)
        self.assertEqual(m.snapshot(1.2)['state'],'INVALID_REPORT')

    def test_bad_strength(self):
        for s in (-.1,1.1,True,float('nan'),float('inf'),'0.5'):
            with self.assertRaises(ValueError):checked_report(report(strength=s))

    def test_bad_distance(self):
        for d in (-1,0,True,None,float('nan')):
            with self.assertRaises(ValueError):checked_report(report(distance_m=d))

    def test_no_near_without_distance(self):
        self.assertEqual(checked_report(report(state='NO_NEAR_CANDIDATE',strength=0.,distance_m=None))['strength'],0)

    def test_unknown_with_strength_rejected(self):
        with self.assertRaises(ValueError):checked_report(report(valid=False,state='FRAME_MISMATCH'))

    def test_known_error_cannot_be_valid(self):
        with self.assertRaises(ValueError):checked_report(report(state='NEAR_FIELD_UNCERTAIN'))

    def test_bad_shape_or_fields(self):
        for text in ('[]','null','{}',report(valid='true'),report(self_mask_enabled='false')):
            with self.assertRaises(ValueError):checked_report(text)

    def test_bad_diagnostics(self):
        for kw in [dict(input_points=-1),dict(input_points=True),dict(raw_nearest_m=float('nan')),dict(processing_ms=-1)]:
            with self.assertRaises(ValueError):checked_report(report(**kw))

    def test_clock_reversal(self):
        m=ViewModel();m.receive(report(),2.)
        self.assertEqual(m.snapshot(1.)['state'],'DISPLAY_STALE')

    def test_unbounded_values_do_not_expand_layout(self):
        for kw in [dict(distance_m=1e100),dict(raw_nearest_m=1e100),dict(input_points=10**100)]:
            with self.assertRaises(ValueError):checked_report(report(**kw))

    def test_report_length_guard(self):
        with self.assertRaises(ValueError):checked_report(' '*100001)


if __name__=='__main__':unittest.main()
