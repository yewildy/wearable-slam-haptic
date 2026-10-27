#!/usr/bin/env python3
"""ROS 接收与发布框架。学习过滤规则时，主要修改 filter_logic.py。"""
from email import message

import numpy as np
import rospy
from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from filter_logic import PointFilter


def main():
    # 一、注册节点。学习话题独立于原来的提醒程序和建图程序。
    rospy.init_node('haptic_learning_filter', anonymous=True)
    input_topic = rospy.get_param('~input_topic', '/livox/lidar')
    output_topic = '/haptic_learning/filtered_points'
    publisher = rospy.Publisher(output_topic, PointCloud2, queue_size=1)

    # 二、只创建一次过滤器，之后每一帧都使用这个对象。
    point_filter = PointFilter()

    def on_cloud(message):
        if message.point_num != len(message.points):
            rospy.logwarn_throttle(1.0, '点数声明不一致，本帧跳过。')
            return
        
        if message.header.frame_id != 'livox_frame':
            rospy.logwarn_throttle(1.0, '需要 livox_frame，收到 %s，本帧跳过。',
                                  message.header.frame_id)
            return


        # 三、把 CustomMsg 中的点变成 N 行 3 列的数组。
        points = np.asarray([(p.x, p.y, p.z) for p in message.points],
                            dtype=float).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
        clean_points = points[valid]


        # 四、调用你写的函数。这是接收框架与过滤规则连接的位置。
        # 1. 拿到原始点的 X, Y, Z (不做复杂的 NaN / 0 点清洗)

       

        # 2. 直接扔进你的算法里
        kept = point_filter.filter(clean_points, message.header.stamp.to_sec())

# 3. 搞定，准备发布或画图！

        # 五、把保留点发布出去。继承原帧的坐标系和时间，不做坐标变换。
        output = point_cloud2.create_cloud_xyz32(message.header, kept.tolist())
        publisher.publish(output)
        rospy.loginfo_throttle(1.0, '对方发布的第 %d 帧 | 原始点=%d，有效点=%d，保留点=%d',
                              message.header.seq, len(points), len(clean_points), len(kept))

    # 六、订阅雷达话题，并持续等待回调。
    rospy.Subscriber(input_topic, CustomMsg, on_cloud, queue_size=1,
                     buff_size=16*1024*1024, tcp_nodelay=True)
    rospy.loginfo('学习节点：%s -> filter_logic.py -> %s', input_topic, output_topic)
    rospy.spin()    


if __name__ == '__main__':
    main()
