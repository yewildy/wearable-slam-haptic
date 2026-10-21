#!/usr/bin/env python3

import rospy
from livox_ros_driver2.msg import CustomMsg, CustomPoint


def main():
    rospy.init_node("haptic_demo_points")
    publisher = rospy.Publisher("/haptic/demo/lidar", CustomMsg, queue_size=1)
    rate = rospy.Rate(1)
    rospy.logwarn("SYNTHETIC DEMO: four artificial points, not a physical LiDAR.")
    while not rospy.is_shutdown():
        message = CustomMsg()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "haptic_demo"
        message.timebase = message.header.stamp.to_nsec()
        for x in (1.0, 2.0, 3.0, 4.0):
            point = CustomPoint()
            point.x = x
            point.reflectivity = 100
            message.points.append(point)
        message.point_num = len(message.points)
        publisher.publish(message)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
