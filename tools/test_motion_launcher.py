import signal,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import start_motion as launch


class Process:
    def __init__(self,node=False,die=False,viewer_wait=False):
        self.node,self.die,self.viewer_wait=node,die,viewer_wait
        self.ended=False;self.signals=[];self.calls=0;self.returncode=0
    def poll(self):
        self.calls+=1
        if self.ended:return 0
        if self.node:return 1 if self.die and self.calls>=2 else None
        return None if self.viewer_wait else 0
    def send_signal(self,s):self.signals.append(s)
    def wait(self,timeout=None):self.ended=True;return 0


class LauncherTests(unittest.TestCase):
    def exercise(self,existing=(),offline=False,die=False,race=False,argv=()):
        commands=[];processes=[]
        def state():
            pubs=list(existing) if not commands else (['/racer'] if race else ['/haptic_motion_owned_123'])
            return [['/haptic/motion_v3/report',pubs]],[],[]
        master=types.SimpleNamespace(getPid=lambda:1,getSystemState=state,
                                     getTopicTypes=lambda:[['/haptic/motion_v3/report','std_msgs/String']])
        if offline:master.getPid=lambda:(_ for _ in ()).throw(OSError())
        graph=types.SimpleNamespace(Master=lambda name:master)
        def popen(command):
            commands.append(command)
            p=Process(node=command[1].endswith('motion_node.py'),die=die,viewer_wait=die)
            processes.append(p);return p
        with patch.dict(sys.modules,{'rosgraph':graph}),patch.object(sys,'argv',['start']+list(argv)),\
             patch.object(launch.os,'getpid',return_value=123),patch.object(launch.subprocess,'Popen',side_effect=popen):
            result=launch.main()
        return result,commands,processes

    def test_owned_node_and_viewer(self):
        result,commands,processes=self.exercise()
        self.assertEqual(result,0);self.assertEqual(len(commands),2)
        self.assertEqual(processes[0].signals,[signal.SIGINT])

    def test_existing_node_only_opens_viewer(self):
        result,commands,processes=self.exercise(['/existing'])
        self.assertEqual(result,0);self.assertEqual(len(commands),1)
        self.assertTrue(commands[0][1].endswith('motion_display.py'))

    def test_offline_and_conflicting_start_nothing(self):
        for args in ({'offline':True},{'existing':['/a','/b']}):
            result,commands,_=self.exercise(**args)
            self.assertEqual(result,2);self.assertFalse(commands)

    def test_owned_calculator_exit_closes_owned_viewer(self):
        result,commands,processes=self.exercise(die=True)
        self.assertEqual(result,2);self.assertEqual(len(commands),2)
        self.assertEqual(processes[1].signals,[signal.SIGINT])

    def test_registration_race_stops_only_owned_node(self):
        result,commands,processes=self.exercise(race=True)
        self.assertEqual(result,2);self.assertEqual(len(commands),1)
        self.assertEqual(processes[0].signals,[signal.SIGINT])

    def test_demo_does_not_need_ros(self):
        with patch.object(sys,'argv',['start','--demo']),patch.object(launch.subprocess,'call',return_value=0) as call:
            self.assertEqual(launch.main(),0)
            self.assertIn('--demo',call.call_args[0][0])

    def test_shell_lf_and_preserved_arguments(self):
        source=Path(__file__).with_name('start_motion.sh').read_bytes()
        self.assertNotIn(b'\r',source);self.assertIn(b'APP_ARGS=("$@")',source)
        self.assertIn(b'set --',source)


if __name__=='__main__':unittest.main()
