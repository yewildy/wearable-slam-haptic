#!/usr/bin/env python3

import math

import rospy
from livox_ros_driver2.msg import CustomMsg
from std_msgs.msg import Float32, String, UInt32


def nearest_distance(points):
    distances = (
        math.hypot(point.x, point.y, point.z)
        for point in points
        if all(math.isfinite(value) for value in (point.x, point.y, point.z))
    )
    # No finite points means unknown, not zero distance or a clear path.
    return min(distances, default=float("nan"))


def main():
    rospy.init_node("haptic_feedback_node")
    publisher = rospy.Publisher("/haptic/status", String, queue_size=1)
    count_publisher = rospy.Publisher("/haptic/point_count", UInt32, queue_size=1)
    distance_publisher = rospy.Publisher("/haptic/nearest_distance", Float32, queue_size=1)
    input_topic = rospy.get_param("~input_topic", "/livox/lidar")

    def on_points(message):
        count = len(message.points)
        count_publisher.publish(UInt32(data=count))
        distance_publisher.publish(Float32(data=nearest_distance(message.points)))
        rospy.loginfo_throttle(1.0, "Received %d points from %s", count, input_topic)

    subscriber = rospy.Subscriber(
        input_topic, CustomMsg, on_points, queue_size=1, buff_size=4 * 1024 * 1024
    )
    rate = rospy.Rate(1)
    status = String(data=(
        "prototype_running; obstacle_detection_not_implemented; "
        "motor_control_disabled"
    ))
    rospy.loginfo("Listening to %s; publishing point counts to /haptic/point_count", input_topic)

    while not rospy.is_shutdown():
        publisher.publish(status)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
