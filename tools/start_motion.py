#!/usr/bin/env python3
"""第三步独立入口；不启动驱动/建图，不结束已有节点，只清理自己创建的进程。"""
import argparse,os,socket,subprocess,sys,time
from pathlib import Path
from start_strength_display import stop_owned


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--viewer-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.demo:return subprocess.call([sys.executable,str(root/'motion_display.py'),'--demo'])
    import rosgraph
    socket.setdefaulttimeout(3.)
    try:
        master=rosgraph.Master('/motion_v3_launcher');master.getPid()
        existing=dict(master.getSystemState()[0]).get('/haptic/motion_v3/report',[])
        types=dict(master.getTopicTypes())
    except Exception:
        print('ROS master unavailable. Start your existing MID-360 launch first.',file=sys.stderr);return 2
    finally:socket.setdefaulttimeout(None)
    if len(existing)>1 or (existing and types.get('/haptic/motion_v3/report')!='std_msgs/String'):
        print('Conflicting stage-three publishers; none stopped.',file=sys.stderr);return 2
    node=viewer=None
    try:
        if not existing and not args.viewer_only:
            name='/haptic_motion_owned_%d'%os.getpid()
            node=subprocess.Popen([sys.executable,str(root/'motion_node.py'),'__name:='+name[1:]])
            deadline=time.monotonic()+10.
            while time.monotonic()<deadline:
                if node.poll() is not None:
                    print('Motion node exited before publishing; no empty viewer opened.',file=sys.stderr);return 2
                publishers=dict(master.getSystemState()[0]).get('/haptic/motion_v3/report',[])
                if publishers==[name]:break
                if publishers:
                    print('Another publisher appeared during startup; only owned child will stop.',file=sys.stderr);return 2
                time.sleep(.1)
            else:
                print('Owned node did not register its report.',file=sys.stderr);return 2
        viewer=subprocess.Popen([sys.executable,str(root/'motion_display.py')])
        while viewer.poll() is None:
            if node is not None and node.poll() is not None:
                print('Owned motion node stopped; closing owned viewer.',file=sys.stderr);return 2
            time.sleep(.1)
        return viewer.returncode
    finally:
        stop_owned(viewer);stop_owned(node)


if __name__=='__main__':sys.exit(main())
