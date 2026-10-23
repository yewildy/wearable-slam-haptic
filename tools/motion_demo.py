"""生成确定性模拟序列；不读真实雷达，不向ROS发布。"""
import json
import numpy as np
from motion_pipeline import MotionEngine


def simulation():
    engine=MotionEngine();reports=[]
    for i in range(340):
        stamp=1+i*.1
        if i<120:distance=1.8
        elif i<130:distance=1.8-(i-119)*.1
        elif i<240:distance=.8
        elif i<260:distance=.8+(i-239)*.05
        else:distance=1.8
        points=np.array([[.1,0,0],[.1,.002,0],[.1,0,.002],[.1,.002,.002],
                         [distance,0,0],[distance,.002,0],[distance,0,.002],[distance,.002,.002]])
        pose=dict(matrix=np.eye(4),frame='demo_world',stamp=stamp)
        report=engine.process(points,'livox_frame',stamp,pose=pose,now=stamp)
        report.update(source_kind='SYNTHETIC_NOT_PHYSICAL',simulation_time_s=i*.1,
                      demonstration_phase='warmup' if i<100 else 'far' if i<120 else 'approach' if i<130 else 'stopped' if i<240 else 'recede' if i<260 else 'far_again')
        reports.append(report)
    return reports


if __name__=='__main__':
    reports=simulation()
    print(json.dumps([reports[i] for i in (110,129,180,230,259,330)],indent=2,allow_nan=False))
