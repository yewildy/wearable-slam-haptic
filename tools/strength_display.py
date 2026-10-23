#!/usr/bin/env python3
"""PyQt只读强度窗口；--demo仅生成窗口内的合成显示，不发布ROS消息。"""
import argparse
import json
import math
import sys
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from strength_view_model import REASONS, ViewModel


class StrengthWindow(QtWidgets.QWidget):
    def __init__(self, model, demo=False):
        super().__init__()
        self.model, self.demo = model, demo
        self.setWindowTitle('距离提醒强度' + (' · 模拟数据' if demo else ''))
        self.resize(900, 520)
        self.setMinimumSize(720, 440)
        self.setFont(QtGui.QFont('Noto Sans CJK SC', 9))
        self.setStyleSheet('QWidget { background: #f7f8fa; color: #20252a; } QLabel { background: transparent; }')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('距离提醒')
        title.setFont(QtGui.QFont('Noto Sans CJK SC', 16, QtGui.QFont.Bold))
        header.addWidget(title)
        header.addStretch()
        self.mode = QtWidgets.QLabel('模拟数据 · 非现场测量' if demo else '实时数据 · 软件强度')
        self.mode.setStyleSheet('color: #9b2f20;' if demo else 'color: #606970;')
        header.addWidget(self.mode)
        layout.addLayout(header)
        self.status = QtWidgets.QLabel('等待数据')
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(38)
        self.status.setContentsMargins(12, 6, 12, 6)
        layout.addWidget(self.status)
        values = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel('提醒强度'))
        self.strength = QtWidgets.QLabel('—')
        self.strength.setFont(QtGui.QFont('Noto Sans CJK SC', 28, QtGui.QFont.Bold))
        self.strength.setMinimumHeight(self.strength.fontMetrics().height()+6)
        left.addWidget(self.strength)
        values.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel('本帧距离'))
        self.distance = QtWidgets.QLabel('—')
        self.distance.setFont(QtGui.QFont('Noto Sans CJK SC', 24, QtGui.QFont.Bold))
        self.distance.setMinimumHeight(self.distance.fontMetrics().height()+6)
        right.addWidget(self.distance)
        values.addLayout(right, 1)
        layout.addLayout(values)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(24)
        layout.addWidget(self.bar)
        self.mask = QtWidgets.QLabel('自身过滤状态：尚未收到')
        self.mask.setWordWrap(True)
        layout.addWidget(self.mask)
        grid = QtWidgets.QGridLayout()
        self.diagnostics = {}
        for i, name in enumerate(('剩余近场点', '滚动排除点（本帧）', '邻域支持点', '输入点数')):
            grid.addWidget(QtWidgets.QLabel(name), i//2, 2*(i%2))
            label = QtWidgets.QLabel('—')
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(label, i//2, 2*(i%2)+1)
            self.diagnostics[name] = label
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setHorizontalSpacing(20)
        layout.addLayout(grid)
        self.detail = QtWidgets.QLabel('距离候选：—')
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet('color: #606970;')
        layout.addWidget(self.detail)
        layout.addStretch(1)
        footer = QtWidgets.QHBoxLayout()
        self.scope = QtWidgets.QLabel('全部可见方向 · 无地面分类 · 马达未接入')
        self.scope.setStyleSheet('color: #606970; font-size: 12px;')
        footer.addWidget(self.scope)
        footer.addStretch()
        self.age = QtWidgets.QLabel('—')
        self.age.setStyleSheet('color: #606970; font-size: 12px;')
        footer.addWidget(self.age)
        layout.addLayout(footer)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        self.refresh()

    def refresh(self):
        r = self.model.snapshot()
        valid = r['valid']
        state = r['state']
        reason = REASONS.get(state, '未知状态')
        if valid:
            value = r['strength']
            color = '#bf3535' if value >= .75 else '#b87713' if value >= .25 else '#26775d'
            self.strength.setText('%d%%' % round(100*value))
            self.bar.setValue(round(1000*value))
            self.distance.setText('%.2f m' % r['distance_m'] if r['distance_m'] is not None else '—')
            self.status.setText(reason)
            background, foreground = ('#fce8e8', '#922b2b') if value >= .75 else ('#fff0d6', '#775013') if value >= .25 else ('#e9f1ed', '#245a47')
            self.status.setStyleSheet('background: '+background+'; color: '+foreground+'; border-radius: 4px;')
        else:
            color = '#aab0b5'
            self.strength.setText('未知')
            self.bar.setValue(0)
            self.distance.setText('—')
            self.status.setText(reason+' · 暂无有效提醒强度')
            self.status.setStyleSheet('background: #fff0d6; color: #775013; border-radius: 4px;')
        self.strength.setStyleSheet('color: '+(color if valid else '#535e66')+';')
        self.bar.setStyleSheet('QProgressBar {border: 0; background: #e1e5e8; border-radius: 3px;} '
                              'QProgressBar::chunk { background: '+color+'; border-radius: 3px; }')
        if r.get('mask_state_known', True):
            self.mask.setText('自身过滤：已应用审核配置（审核声明）' if r['self_mask_enabled'] else '自身过滤：未启用')
            if 'rolling_removed_points' in r:
                self.mask.setText('滚动10秒 / 50 cm · %s · 原型测试，屏蔽区不代表安全' % ('已就绪' if r.get('rolling_ready') else '统计中'))
        else:
            self.mask.setText('自身过滤状态：暂无新数据')
        self.mask.setStyleSheet('color: #795313;' if r.get('self_mask_enabled') else 'color: #606970;')
        keys = ('near_untrusted_points', 'rolling_removed_points', 'supported_points', 'original_input_points')
        stale = state in ('DISPLAY_STALE', 'INVALID_REPORT', 'WAITING_FOR_REPORT')
        for label, key in zip(self.diagnostics.values(), keys):
            label.setText(str(r.get(key)) if not stale and r.get(key) is not None else '—')
        candidate = r.get('candidate_distance_m')
        raw = r.get('raw_nearest_m')
        parts = ['距离候选（仅诊断）：'+('%.3f m' % candidate if candidate is not None else '—'),
                 '剩余最近回波：'+('%.3f m' % raw if raw is not None else '—')]
        if not stale and r.get('candidate_strength') is not None:
            parts.append('候选强度（非控制值）：%d%%' % round(100*r['candidate_strength']))
        self.detail.setText('    '.join(parts))
        self.age.setText('报告 %.1f s 前' % r['age'] if r.get('age') is not None else '等待报告')


def demo_report(distance):
    # 不读现场数据、不发布ROS；与第二步默认曲线一致的显示样例。
    strength = 1. if distance <= .5 else .5+.5*(1-distance)/.5 if distance <= 1 else max(0., .5*(2-distance))
    return dict(valid=True,state='DISTANCE_CANDIDATE' if distance<=2 else 'NO_NEAR_CANDIDATE',
                strength=strength,distance_m=distance,self_mask_enabled=False,
                candidate_distance_m=distance,raw_nearest_m=distance,input_points=20000,
                near_untrusted_points=0,self_removed_points=0,supported_points=1200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    app = QtWidgets.QApplication([sys.argv[0]])
    model = ViewModel()
    if not args.demo:
        import rospy
        from std_msgs.msg import String
        rospy.init_node('haptic_strength_display', anonymous=True, disable_signals=True)
        subscriber = rospy.Subscriber('/haptic/distance_v2/report', String,
                                      lambda msg:model.receive(msg.data), queue_size=1)
        app.aboutToQuit.connect(lambda:rospy.signal_shutdown('Display closed'))
    window = StrengthWindow(model, args.demo)
    if args.demo:
        started = time.monotonic()
        timer = QtCore.QTimer(window)
        timer.timeout.connect(lambda:model.receive(json.dumps(demo_report(1.25+.85*math.sin(time.monotonic()-started)))))
        timer.start(100)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
