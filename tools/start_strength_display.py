#!/usr/bin/env python3
"""检查已有ROS，按需启动自己的距离节点；关闭界面只清理自己启动的进程。"""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--viewer-only', action='store_true')
    parser.add_argument('--reviewed-profile', type=Path)
    parser.add_argument('--apply-self-mask', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.demo:
        if args.reviewed_profile or args.apply_self_mask:
            parser.error('Demo does not apply any real mask')
        return subprocess.call([sys.executable,str(root/'strength_display.py'),'--demo'])
    if args.apply_self_mask and not args.reviewed_profile:
        parser.error('Applying a mask requires an explicitly reviewed profile')
    if args.reviewed_profile and not args.reviewed_profile.is_file():
        parser.error('Reviewed profile does not exist')
    import rosgraph
    socket.setdefaulttimeout(3.)
    try:
        master = rosgraph.Master('/haptic_strength_launcher')
        master.getPid()
        publishers, _, _ = master.getSystemState()
        current = dict(publishers).get('/haptic/distance_v2/report', [])
        types = dict(master.getTopicTypes())
    except Exception:
        print('ROS is not running/reachable. Start your existing MID-360 launch first; then run this command again.',file=sys.stderr)
        return 2
    finally:
        socket.setdefaulttimeout(None)
    if len(current)>1 or (current and types.get('/haptic/distance_v2/report')!='std_msgs/String'):
        print('Conflicting distance report publishers. No process was stopped.',file=sys.stderr)
        return 2
    if current and (args.reviewed_profile or args.apply_self_mask):
        print('A distance node already exists. Close it yourself before changing its mask settings.',file=sys.stderr)
        return 2
    node = viewer = None
    try:
        if not current and not args.viewer_only:
            command=[sys.executable,str(root/'distance_node.py'),
                     '__name:=haptic_distance_display_%d' % os.getpid()]
            if args.reviewed_profile:
                command.append('_reviewed_profile:='+str(args.reviewed_profile.resolve()))
            if args.apply_self_mask:
                command.append('_apply_self_mask:=true')
            node=subprocess.Popen(command)
        viewer=subprocess.Popen([sys.executable,str(root/'strength_display.py')])
        return viewer.wait()
    finally:
        stop_owned(viewer)
        stop_owned(node)


if __name__=='__main__':
    sys.exit(main())
