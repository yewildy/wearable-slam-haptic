import argparse
import collections
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

import rospy
import yaml
from diagnostic_msgs.msg import DiagnosticArray

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
args=parser.parse_args()
root=args.directory
port=11326
with socket.socket() as probe: probe.bind(('127.0.0.1',port))
os.environ['ROS_MASTER_URI']=f'http://127.0.0.1:{port}'
os.environ['ROS_IP']='127.0.0.1'
os.environ.pop('ROS_HOSTNAME',None)
profile=yaml.safe_load((root/'candidate_mask.yaml').read_text())
assert not profile['self_identity_confirmed']
profile['self_mask_enabled']=True
trial=root/'replay_only_mask.yaml'
trial.write_text(yaml.safe_dump(profile))
rows=[]
def receive(message):
    if message.status and message.header.frame_id=='livox_frame':
        status=message.status[0]
        rows.append((status.message,{v.key:v.value for v in status.values}))
launch=player=None
with (root/'replay_candidate.log').open('w') as log:
    try:
        launch=subprocess.Popen(['roslaunch','-p',str(port),'haptic_feedback','pcl_proximity.launch',
            'self_mask_config:='+str(trial)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(100):
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.1):break
            except OSError:time.sleep(.1)
        rospy.init_node('replay_candidate_report',disable_signals=True)
        subscriber=rospy.Subscriber('/haptic/proximity/diagnostics',DiagnosticArray,receive,queue_size=100)
        for _ in range(100):
            if subscriber.get_num_connections():break
            time.sleep(.05)
        assert subscriber.get_num_connections()
        player=subprocess.Popen(['rosbag','play','--wait-for-subscribers','-r','1',str(root/'raw.bag'),
            '--topics','/livox/lidar'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        assert player.wait(timeout=45)==0
        time.sleep(.2)
        assert rows,'No frame results'
        report={'replay_only':True,'self_identity_confirmed':False,'frames_processed':len(rows),
            'states':dict(collections.Counter(s for s,_ in rows)), 'mask_box':profile['self_boxes']}
        for key in ['distance_m','self_masked_points','near_untrusted_points','processing_ms']:
            values=[float(v[key]) for _,v in rows if key in v and math.isfinite(float(v[key]))]
            if values:report[key]={'min':min(values),'mean':statistics.mean(values),'max':max(values)}
    finally:
        if rospy.core.is_initialized():rospy.signal_shutdown('Replay complete')
        for process in [player,launch]:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
                try:process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5)
with socket.socket() as probe:assert probe.connect_ex(('127.0.0.1',port))!=0
report['temporary_master_stopped']=True
(root/'replay_candidate_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
