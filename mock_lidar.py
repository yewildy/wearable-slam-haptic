#!/usr/bin/env python3
"""学习用模拟雷达：每秒在独立话题发布一帧四个点，不启动硬件。"""
import rospy
from livox_ros_driver2.msg import CustomMsg, CustomPoint


def make_message(stamp, sequence):
    # 一、帧信息：消息类型、坐标系与真实驱动一致；时间来自当前ROS时钟。
    message = CustomMsg()
    message.header.seq = sequence
    message.header.stamp = stamp
    message.header.frame_id = 'livox_frame'
    message.timebase = stamp.to_nsec()
    message.lidar_id = 0
    message.rsvd = [0, 0, 0]

    # 二、四个模拟点，单位是米。其他字段是人为设置，不是传感器测量。
    point_configs = [
        # --- 近距离点 (0.5m 以内) ---
        {"desc": "近点-正前方", "coords": (0.20,  0.10,  0.05)},
        {"desc": "近点-左前方", "coords": (0.35, -0.15, -0.02)},
        {"desc": "近点-右上方", "coords": (0.45,  0.05,  0.12)},
        
        # --- 远距离点 (0.5m 以外) ---
        {"desc": "远点-正前方", "coords": (0.80,  0.25, -0.10)},
        {"desc": "远点-左侧方", "coords": (1.20, -0.40,  0.30)},
        {"desc": "远点-高处点", "coords": (2.50,  0.80,  0.50)},
    ]

    # 2. 统一的对象构造与赋值流程
    for index, config in enumerate(point_configs):
        x, y, z = config["coords"]
        
        point = CustomPoint()
        point.offset_time = index * 1000  # 相对 timebase 的时间偏移 (ns)
        point.x, point.y, point.z = x, y, z
        point.reflectivity = 100          # 默认反射率
        point.tag = 0                     # 默认点属性 Tag
        point.line = 0                    # 激光线束 ID
        
        message.points.append(point)

    message.point_num = len(message.points)
    return message


def main():
    # --- 1. 配置参数 (集中管理，避免硬编码) ---
    NODE_NAME = 'haptic_learning_mock_lidar'
    TOPIC_NAME = '/haptic_learning/mock_lidar'
    PUBLISH_RATE_HZ = 1.0  # 发送频率：1 Hz (每秒 1 帧点云)
    
    # --- 2. 初始化 ROS 节点与发布者 ---
    rospy.init_node(NODE_NAME, anonymous=True)
    publisher = rospy.Publisher(TOPIC_NAME, CustomMsg, queue_size=1)
    rate = rospy.Rate(PUBLISH_RATE_HZ)
    
    rospy.loginfo("模拟雷达节点已启动，发布话题: %s", TOPIC_NAME)

    sequence = 0
    
    # --- 3. 消息发布循环 ---
    while not rospy.is_shutdown():
        current_time = rospy.Time.now()
        
        # 生成模拟消息并发布
        mock_msg = make_message(current_time, sequence)
        publisher.publish(mock_msg)
        
        # 动态获取当前帧的点数，使日志与数据真实同步
        point_count = len(mock_msg.points)
        rospy.loginfo("成功发布模拟帧 [Seq: %d] | 包含点数: %d", sequence, point_count)
        
        # 序列号递增 (防止溢出，按 uint32 范围循环)
        sequence = (sequence + 1) % (2 ** 32)
        
        # 保持指定的发布频率
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
