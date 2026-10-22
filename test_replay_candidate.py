import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

script = Path(__file__).parent/'replay_candidate.py'
if not script.exists(): script=Path.home()/'haptic_ws/tools/replay_candidate.py'
spec=importlib.util.spec_from_file_location('replay_candidate',script)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def profile():
    return {'self_identity_confirmed':False,'self_mask_enabled':False,'self_mask_frame':'fixture_frame',
            'self_boxes':[{'min':[-.4,-.1,-.1],'max':[-.05,.1,.1]}]}


@patch.object(module.signal,'SIGKILL',9,create=True)
class ReplayTests(unittest.TestCase):
    def test_trial_copy_does_not_mutate_candidate(self):
        original=profile(); result=module.trial_profile(original)
        self.assertTrue(result['self_mask_enabled']);self.assertFalse(original['self_mask_enabled'])
        result['self_boxes'][0]['min'][0]=-1
        self.assertEqual(original['self_boxes'][0]['min'][0],-.4)

    def test_confirmed_profile_is_not_silently_repurposed(self):
        value=profile();value['self_identity_confirmed']=True
        with self.assertRaises(ValueError):module.trial_profile(value)

    def test_empty_frame_and_boxes_rejected(self):
        for key,value in [('self_mask_frame',''),('self_boxes',[])]:
            p=profile();p[key]=value
            with self.assertRaises(ValueError):module.trial_profile(p)

    def test_nonfinite_boolean_and_inverted_bounds_rejected(self):
        for value in [float('nan'),float('inf'),True,'x',1001]:
            p=profile();p['self_boxes'][0]['min'][0]=value
            with self.assertRaises(ValueError):module.trial_profile(p)
        p=profile();p['self_boxes'][0]['min'][0]=1
        with self.assertRaises(ValueError):module.trial_profile(p)

    def test_bad_dimensions_and_missing_keys_rejected(self):
        for value in [None,{'min':[0,0],'max':[1,1,1]}, {'min':[0,0,0]}]:
            p=profile();p['self_boxes']=[value]
            with self.assertRaises(ValueError):module.trial_profile(p)

    def test_dead_launch_fails_without_waiting_for_ros(self):
        with self.assertRaisesRegex(RuntimeError,'exited'):
            module.wait_ready(lambda:False,SimpleNamespace(poll=lambda:1),1,'Master')

    def test_readiness_deadline(self):
        with patch.object(module.time,'monotonic',side_effect=[0,0,2]),patch.object(module.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'timed out'):
                module.wait_ready(lambda:False,SimpleNamespace(poll=lambda:None),1,'Master')

    def test_ready_returns(self):
        module.wait_ready(lambda:True,SimpleNamespace(poll=lambda:None),1,'Master')

    def test_generic_frame_summary_and_missing_frames(self):
        report=module.summarize([('fixture_frame','MEDIUM',{'distance_m':'.8','processing_ms':'2'})],2,'fixture_frame')
        self.assertEqual(report['frame'],'fixture_frame')
        self.assertFalse(report['complete_frame_count'])
        self.assertEqual(report['distance_m']['mean'],.8)

    def test_mismatch_is_visible_not_dropped(self):
        report=module.summarize([('other','SELF_MASK_FRAME_MISMATCH',{'distance_m':'nan'})],1,'fixture_frame')
        self.assertEqual(report['frame_mismatch_outputs'],1)
        self.assertNotIn('distance_m',report)

    def test_cleanup_uses_only_owned_process_group(self):
        process=Mock(pid=321)
        with patch.object(module.os,'killpg',create=True) as kill:
            module.stop_owned_group(process)
            kill.assert_called_once_with(321,signal.SIGINT)
            process.wait.assert_called_once_with(timeout=12)

    def test_cleanup_escalates_boundedly(self):
        process=Mock(pid=321)
        process.wait.side_effect=[subprocess.TimeoutExpired('fixture',12),None]
        with patch.object(module.os,'killpg',create=True) as kill:
            module.stop_owned_group(process)
            self.assertEqual([c.args[1] for c in kill.call_args_list],[signal.SIGINT,signal.SIGTERM])

    def test_cleanup_already_gone(self):
        process=Mock(pid=321)
        with patch.object(module.os,'killpg',side_effect=ProcessLookupError,create=True):
            module.stop_owned_group(process)
        process.wait.assert_not_called()

    def test_cleanup_final_kill_is_bounded(self):
        process=Mock(pid=321)
        process.wait.side_effect=[subprocess.TimeoutExpired('fixture',12),subprocess.TimeoutExpired('fixture',5),None]
        with patch.object(module.os,'killpg',create=True) as kill:
            module.stop_owned_group(process)
            self.assertEqual([c.args[1] for c in kill.call_args_list],[signal.SIGINT,signal.SIGTERM,9])

    def test_missing_raw_bag_rejected_before_ros(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'missing'):
                module.inspect_bag(Path(directory)/'raw.bag','fixture_frame')

    def inspect_fixture(self,topic,frame='fixture_frame'):
        bag=Mock()
        bag.__enter__=Mock(return_value=bag)
        bag.__exit__=Mock(return_value=False)
        bag.get_type_and_topic_info.return_value=SimpleNamespace(topics=topic)
        bag.read_messages.return_value=iter([('/livox/lidar',SimpleNamespace(header=SimpleNamespace(frame_id=frame)),None)])
        bag.get_end_time.return_value=2
        bag.get_start_time.return_value=1
        with tempfile.NamedTemporaryFile() as file, patch.dict(sys.modules,{'rosbag':SimpleNamespace(Bag=lambda *args:bag)}):
            return module.inspect_bag(Path(file.name),'fixture_frame')

    def test_missing_topic_rejected(self):
        with self.assertRaisesRegex(ValueError,'No recorded'):
            self.inspect_fixture({})

    def test_wrong_message_type_rejected(self):
        with self.assertRaisesRegex(ValueError,'CustomMsg'):
            self.inspect_fixture({'/livox/lidar':SimpleNamespace(message_count=1,msg_type='sensor_msgs/PointCloud2')})

    def test_recorded_frame_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'does not match'):
            self.inspect_fixture({'/livox/lidar':SimpleNamespace(message_count=1,msg_type='livox_ros_driver2/CustomMsg')},'other')

    def test_valid_bag_metadata(self):
        value=self.inspect_fixture({'/livox/lidar':SimpleNamespace(message_count=5,msg_type='livox_ros_driver2/CustomMsg')})
        self.assertEqual(value,{'frames_expected':5,'bag_duration_seconds':1})


if __name__=='__main__': unittest.main(verbosity=2)
