#!/usr/bin/env python3
"""独立ROS1第三步节点。仅订阅已有雷达/TF，发布motion_v3，不控制硬件。"""
import json
import threading
import time
import numpy as np
from motion_pipeline import MotionEngine
from motion_tracking import MotionSettings
from motion_pose import PoseProvider


def main():
    import rospy,tf
    from livox_ros_driver2.msg import CustomMsg
    from std_msgs.msg import String,Float32
    rospy.init_node('haptic_motion_v3')
    settings={name:rospy.get_param('~motion_'+name,field.default) for name,field in MotionSettings.__dataclass_fields__.items()}
    engine=MotionEngine(MotionSettings(**settings));poses=PoseProvider(rospy,tf)
    prefix='/haptic/motion_v3/'
    pubs={name:rospy.Publisher(prefix+name,kind,queue_size=1) for name,kind in
          [('report',String),('state',String),('strength',Float32),('distance',Float32)]}
    lock=threading.Lock();started=[None]

    def publish(report):
        report.update(algorithm_mode='motion_v3',motor_control=False)
        pubs['report'].publish(json.dumps(report,allow_nan=False))
        pubs['state'].publish(report['state'])
        for key,field in [('strength','strength'),('distance','distance_m')]:
            pubs[key].publish(float('nan') if report.get(field) is None else report[field])

    def callback(message):
        with lock:
            start=time.monotonic();started[0]=start;age=rospy.Time.now().to_sec()-message.header.stamp.to_sec()
            try:
                xyz=np.asarray([(p.x,p.y,p.z) for p in message.points],dtype=float).reshape(-1,3)
                pose,source=poses.at(message.header.frame_id,message.header.stamp)
                report=engine.process(xyz,message.header.frame_id,message.header.stamp.to_sec(),age,
                                      int(message.point_num),pose,start)
                report['pose_source']=source
            except (ValueError,TypeError,KeyError,RuntimeError) as error:
                rospy.logerr_throttle(2.,'Motion input rejected: %s',error)
                report=engine.unknown('INPUT_OR_PROFILE_ERROR')
            elapsed=time.monotonic()-start
            if elapsed>.5:report=engine.unknown('PROCESSING_TOO_SLOW')
            elif report['valid'] and rospy.Time.now().to_sec()-message.header.stamp.to_sec()>.5:
                report=engine.unknown('STALE_AT_PUBLICATION')
            report.update(source_age_s=age,input_stamp=message.header.stamp.to_sec(),
                          input_frame=message.header.frame_id,processing_ms=elapsed*1000)
            publish(report);started[0]=None

    def watchdog():
        while not rospy.is_shutdown():
            if lock.acquire(False):
                try:
                    if engine.base.watchdog() is not None:publish(engine.unknown('STALE_OR_NO_DATA'))
                finally:lock.release()
            elif started[0] is not None and time.monotonic()-started[0]>.5:
                report=engine.base._report('PROCESSING_TOO_SLOW')
                report.update(motion_valid=False,tracks=[],pose_state='UNAVAILABLE');publish(report)
            time.sleep(.1)

    subscriber=rospy.Subscriber(rospy.get_param('~input_topic','/livox/lidar'),CustomMsg,callback,
                                queue_size=1,buff_size=16*1024*1024,tcp_nodelay=True)
    threading.Thread(target=watchdog,daemon=True).start()
    rospy.logwarn('STAGE3 PROTOTYPE: unresolved near returns stay strong; coefficients not calibrated; no motor')
    rospy.spin()


if __name__=='__main__':main()
