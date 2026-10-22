import argparse
import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys

import rosbag
import rospy
import yaml
from livox_ros_driver2.msg import CustomMsg, CustomPoint

parser=argparse.ArgumentParser()
parser.add_argument('--real-directory',type=Path)
args=parser.parse_args()
script=Path(__file__).parent/'replay_candidate.py'
root=Path.home()/'haptic_ws/logs/20261022'/('replay_integration_'+datetime.datetime.now().strftime('%H%M%S_%f'))
fixture=root/'fixture'
fixture.mkdir(parents=True,exist_ok=False)
frame='fixture_frame'
candidate={'self_mask_enabled':False,'self_identity_confirmed':False,'self_mask_frame':frame,
           'self_boxes':[{'min':[-.4,-.1,-.1],'max':[-.05,.1,.1]}]}
(fixture/'candidate_mask.yaml').write_text(yaml.safe_dump(candidate))
with rosbag.Bag(str(fixture/'raw.bag'),'w') as bag:
    for i in range(5):
        message=CustomMsg()
        message.header.frame_id=frame
        message.header.stamp=rospy.Time.from_sec(1+i*.1)
        for x in [-.2,.8]:
            for j in range(6):
                point=CustomPoint();point.x=x;point.y=j*.01;point.z=0
                message.points.append(point)
        message.point_num=len(message.points)
        bag.write('/livox/lidar',message,t=message.header.stamp)


def replay(directory,label):
    result=subprocess.run([sys.executable,str(script),str(directory)],stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,timeout=100,text=True)
    (root/(label+'.log')).write_text(result.stdout)
    assert result.returncode==0,result.stdout
    saved=next(line.split('=',1)[1] for line in result.stdout.splitlines() if line.startswith('SAVED_DIRECTORY='))
    report=json.loads((Path(saved)/'replay_candidate_report.json').read_text())
    assert report['temporary_master_stopped']
    return report


synthetic=replay(fixture,'synthetic')
assert synthetic['frames_processed']==5 and synthetic['frames_expected']==5
assert synthetic['frame']==frame and synthetic['frame_mismatch_outputs']==0
assert synthetic['states']=={'MEDIUM':5}
assert abs(synthetic['distance_m']['mean']-.8)<1e-5
assert yaml.safe_load((fixture/'candidate_mask.yaml').read_text())['self_mask_enabled'] is False
with socket.socket() as occupied:
    occupied.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    occupied.bind(('127.0.0.1',11326))
    occupied.listen(1)
    before=set(fixture.glob('replay_*'))
    result=subprocess.run([sys.executable,str(script),str(fixture)],stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,timeout=15,text=True)
    assert result.returncode!=0
    assert set(fixture.glob('replay_*'))==before,'A run started on an occupied port'

report={'synthetic_frame_name':frame,'synthetic_frames_processed':5,
        'synthetic_range_correct':True,'occupied_port_refused_without_starting':True,
        'candidate_input_unchanged':True,'temporary_master_stopped':True,
        'live_sensor_used':False,'physical_motor_tested':False}
if args.real_directory:
    real=replay(args.real_directory,'saved_real')
    assert real['frames_expected']==141 and real['frames_processed']==141
    assert real['frame_mismatch_outputs']==0 and real['complete_frame_count']
    report['saved_real_bag_frames_processed']=141
    report['saved_real_states']=real['states']
    report['saved_real_range_m']=real.get('distance_m')
    report['known_target_distance_reference']=False
(root/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print('VERIFICATION_FILE='+str(root/'verification.json'))
