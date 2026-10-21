cmake_minimum_required(VERSION 3.0.2)
project(haptic_feedback)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/haptic_feedback_node.py
  scripts/demo_livox_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
