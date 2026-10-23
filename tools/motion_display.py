#!/usr/bin/env python3
"""第三步只读窗口；默认读取独立motion_v3话题，不替换第二步窗口。"""
import argparse,json,sys
from PyQt5 import QtCore,QtWidgets
from strength_display import StrengthWindow
from strength_view_model import ViewModel


class MotionWindow(StrengthWindow):
    def __init__(self,model,demo=False):
        super().__init__(model,demo)
        self.setWindowTitle('第三步运动融合提醒 · 原型'+(' · 模拟' if demo else ''))
        self.resize(900,560)
        self.mode.setText('模拟回放 · 非现场测量' if demo else '第三步 · 运动融合原型')
        for label in self.findChildren(QtWidgets.QLabel):
            if label.text()=='本帧距离':label.setText('主提醒目标距离')
        self.motion_detail=QtWidgets.QLabel('接近速度：等待数据')
        self.motion_detail.setWordWrap(True)
        self.motion_detail.setStyleSheet('color: #4f5964;')
        self.layout().insertWidget(6,self.motion_detail)
        self.refresh()

    def refresh(self):
        super().refresh()
        if not hasattr(self,'motion_detail'):return
        r=self.model.snapshot()
        usable=r['valid'] and r.get('motion_valid',False)
        speed=r.get('closing_speed_mps') if usable else None
        ttc=r.get('ttc_s') if usable else None
        factor=r.get('dwell_factor') if usable else None
        text='接近速度：'+('%.2f m/s'%speed if speed is not None else '未可靠估计')
        text+='    TTC：'+('%.2f s'%ttc if ttc is not None else '—')
        text+='    停留系数：'+('%.2f'%factor if factor is not None else '—')
        state=r.get('pose_state','UNAVAILABLE') if r['valid'] else 'UNAVAILABLE'
        text+='\n位姿：'+('仅相对距离，未补偿自身移动' if state=='RELATIVE_ONLY_NO_POSE' else
                        '世界坐标补偿（继承现有估计）' if state=='WORLD_COMPENSATED' else state)
        if usable:text+='    旧背景：'+('已见过' if r.get('familiar_background') else '未确认')
        if r.get('nearest_distance_m') is not None and usable:
            text+='    最近有效距离：%.2f m'%r['nearest_distance_m']
        text+='\n孤立点排除（本帧）：%d（不计入自身排除）'%r.get('isolated_removed_points',0)
        self.motion_detail.setText(text)
        if r.get('algorithm_mode')=='motion_v3' and r['valid']:
            if r.get('motion_override')=='UNTRUSTED_NEAR_FIELD_NOT_DECAYED':
                self.status.setText('近场残留仍强提醒 · 不使用停留衰减隐藏它')
            elif not r.get('motion_valid'):
                self.status.setText('运动估计不可用 · 保留第二步距离提醒')
            else:
                self.status.setText('运动融合提醒' if state=='WORLD_COMPENSATED' else '相对接近提醒 · 停留衰减未启用')
        if r['valid'] and r.get('distance_only_strength') is not None:
            self.detail.setText('纯距离对照：%d%%    运动融合结果以上方强度为准' % round(100*r['distance_only_strength']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true')
    args=parser.parse_args()
    app=QtWidgets.QApplication([]);model=ViewModel()
    if args.demo:
        from motion_demo import simulation
        reports=simulation();position=[0]
        timer=QtCore.QTimer();timer.setInterval(100)
        def next_frame():
            model.receive(json.dumps(reports[position[0]],allow_nan=False))
            position[0]=(position[0]+1)%len(reports)
        timer.timeout.connect(next_frame);next_frame();timer.start()
    else:
        import rospy
        from std_msgs.msg import String
        rospy.init_node('haptic_motion_display',anonymous=True,disable_signals=True)
        subscription=rospy.Subscriber('/haptic/motion_v3/report',String,lambda m:model.receive(m.data),queue_size=1)
        app.aboutToQuit.connect(lambda:rospy.signal_shutdown('Own display closed'))
    window=MotionWindow(model,args.demo);window.show()
    return app.exec_()


if __name__=='__main__':sys.exit(main())
