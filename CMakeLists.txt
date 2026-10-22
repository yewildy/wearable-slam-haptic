cmake_minimum_required(VERSION 3.0.2)
project(haptic_feedback)

find_package(catkin REQUIRED COMPONENTS rospy roscpp std_msgs diagnostic_msgs livox_ros_driver2)
find_package(PCL 1.10 REQUIRED COMPONENTS common filters search)
catkin_package(CATKIN_DEPENDS rospy roscpp std_msgs diagnostic_msgs livox_ros_driver2)
include_directories(${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pcl_proximity_node src/pcl_proximity_node.cpp)
target_compile_features(pcl_proximity_node PRIVATE cxx_std_14)
target_link_libraries(pcl_proximity_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(pcl_proximity_node ${catkin_EXPORTED_TARGETS})
install(TARGETS pcl_proximity_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS
  scripts/haptic_feedback_node.py
  scripts/demo_livox_publisher.py
  scripts/capture_near_cloud.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
