import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import rospy
from diagnostic_msgs.msg import DiagnosticArray
from livox_ros_driver2.msg import CustomMsg, CustomPoint
import yaml

root = Path.home()/'haptic_ws/logs/20261022/self_mask_tests'
root.mkdir(parents=True,exist_ok=True)
port=11325
with socket.socket() as probe: probe.bind(('127.0.0.1',port))
os.environ['ROS_MASTER_URI']=f'http://127.0.0.1:{port}'
os.environ['ROS_IP']='127.0.0.1'
os.environ.pop('ROS_HOSTNAME',None)
box={'min':[-.4,-.1,-.1], 'max':[-.05,.1,.1]}
profile=root/'synthetic.yaml'
profile.write_text(yaml.safe_dump({'self_mask_enabled':True,'self_mask_frame':'mask_test','self_boxes':[box,box]}))
messages={}
results=[]
def receive(message): messages[message.header.stamp.to_nsec()]=message
def cluster(x): return [(x,i*.01,0) for i in range(6)]
def case(name,points,state,frame='mask_test',**expected):
    message=CustomMsg()
    message.header.frame_id=frame
    for xyz in points:
        p=CustomPoint(); p.x,p.y,p.z=xyz; message.points.append(p)
    message.point_num=len(message.points)
    for _ in range(4):
        message.header.stamp=rospy.Time.now()
        publisher.publish(message)
        time.sleep(.1)
    stamp=message.header.stamp.to_nsec()
    deadline=time.monotonic()+3
    while stamp not in messages and time.monotonic()<deadline: time.sleep(.01)
    assert stamp in messages,name
    status=messages[stamp].status[0]
    values={v.key:v.value for v in status.values}
    assert status.message==state,(name,status.message,state)
    for key,value in expected.items(): assert values[key]==str(value),(name,key,values[key],value)
    assert values['self_mask_enabled']=='true'
    results.append({'case':name,'state':status.message,'level':values['level'],
                    'self_masked_points':values.get('self_masked_points')})
    return values

process=None
with (root/'launch.log').open('w') as log:
    try:
        process=subprocess.Popen(['roslaunch','-p',str(port),'haptic_feedback','pcl_proximity.launch',
            'input_topic:=/haptic/test/cloud','self_mask_config:='+str(profile)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(100):
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.1): break
            except OSError: time.sleep(.1)
        rospy.init_node('test_self_mask',disable_signals=True)
        sub=rospy.Subscriber('/haptic/proximity/diagnostics',DiagnosticArray,receive,queue_size=100)
        publisher=rospy.Publisher('/haptic/test/cloud',CustomMsg,queue_size=1)
        for _ in range(100):
            if publisher.get_num_connections(): break
            time.sleep(.05)
        assert publisher.get_num_connections()
        values=case('self_and_external',cluster(-.25)+cluster(-.1)+cluster(.8)+[(0,0,0),(float('nan'),0,0)],
            'MEDIUM',level=2,self_masked_points=12,zero_points=1,nonfinite_points=1,near_untrusted_points=0)
        assert abs(float(values['distance_m'])-.8)<1e-5
        case('self_only',cluster(-.25),'NO_SUPPORTED_POINTS',level='UNKNOWN',self_masked_points=6)
        case('mask_boundary',cluster(-.05),'NO_SUPPORTED_POINTS',self_masked_points=6)
        case('outside_mask',cluster(-.401),'STRONG',level=3,self_masked_points=0)
        case('external_near_uncertain',cluster(.1),'NEAR_FIELD_UNCERTAIN',level='UNKNOWN',self_masked_points=0)
        case('frame_mismatch',cluster(.8),'SELF_MASK_FRAME_MISMATCH',frame='wrong_frame',level='UNKNOWN')
        case('recovery',cluster(.8),'MEDIUM',level=2,self_masked_points=0)
        for label,params in [('empty',{'self_mask_enabled':True,'self_boxes':[]}),
                             ('invalid',{'self_mask_enabled':True,'self_boxes':[{'min':[1,0,0],'max':[0,1,1]}]})]:
            path=root/(label+'.yaml'); path.write_text(yaml.safe_dump(params))
            name='invalid_mask_'+label
            for key,value in params.items(): rospy.set_param('/'+name+'/'+key,value)
            invalid=subprocess.run(['rosrun','haptic_feedback','pcl_proximity_node','__name:='+name],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10)
            assert invalid.returncode!=0,'Invalid mask was accepted'
        report={'synthetic_only':True,'cases':results,'overlapping_boxes_no_double_count':True,
                'invalid_mask_parameters_rejected':True,'physical_motor_tested':False}
    finally:
        if rospy.core.is_initialized(): rospy.signal_shutdown('Complete')
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try: process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5)
with socket.socket() as probe: assert probe.connect_ex(('127.0.0.1',port))!=0
report['temporary_master_stopped']=True
(root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
