import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import start_strength_display as launch


class FakeProcess:
    def __init__(self):self.ended=False;self.signals=[]
    def poll(self):return 0 if self.ended else None
    def wait(self,timeout=None):self.ended=True;return 0
    def send_signal(self,s):self.signals.append(s)


class LauncherTests(unittest.TestCase):
    def test_shell_script_uses_unix_newlines(self):
        data=Path(__file__).with_name('start_strength_display.sh').read_bytes()
        self.assertNotIn(b'\r',data)

    def exercise(self, publishers, argv=(), failed=False):
        master=types.SimpleNamespace(getPid=lambda:1,getSystemState=lambda:([['/haptic/distance_v2/report',publishers]],[],[]),
                                     getTopicTypes=lambda:[['/haptic/distance_v2/report','std_msgs/String']])
        if failed:master.getPid=lambda:(_ for _ in ()).throw(OSError('offline'))
        graph=types.SimpleNamespace(Master=lambda name:master)
        commands=[];processes=[]
        def popen(command):
            commands.append(command);p=FakeProcess();processes.append(p);return p
        with patch.dict(sys.modules,{'rosgraph':graph}),patch.object(sys,'argv',['start']+list(argv)),patch.object(launch.subprocess,'Popen',side_effect=popen):
            status=launch.main()
        return status,commands,processes

    def test_existing_publisher_not_stopped(self):
        code,commands,processes=self.exercise(['/existing_distance_node'])
        self.assertEqual(code,0);self.assertEqual(len(commands),1)
        self.assertTrue(commands[0][1].endswith('strength_display.py'))
        self.assertEqual(processes[0].signals,[])

    def test_only_owned_node_stopped(self):
        code,commands,processes=self.exercise([])
        self.assertEqual(code,0);self.assertEqual(len(commands),2)
        self.assertTrue(commands[0][1].endswith('distance_node.py'))
        self.assertEqual(processes[0].signals,[launch.signal.SIGINT])

    def test_no_master_no_process(self):
        code,commands,_=self.exercise([],failed=True)
        self.assertEqual(code,2);self.assertFalse(commands)

    def test_conflicts_no_process(self):
        code,commands,_=self.exercise(['/a','/b'])
        self.assertEqual(code,2);self.assertFalse(commands)

    def test_viewer_only(self):
        code,commands,_=self.exercise([],argv=['--viewer-only'])
        self.assertEqual(code,0);self.assertEqual(len(commands),1)


if __name__=='__main__':unittest.main()
