wearable-slam-haptic
===================

我正在现有的Jetson、MID-360和Point-LIO装置上，逐步增加障碍感知与震动提醒。
这个仓库当前是ROS 1话题通信原型，不是已经完成的避障或安全设备。

当前功能
--------
订阅Livox驱动的CustomMsg点云，发布每一帧的点数。
同时保留1 Hz的/haptic/status开发状态消息。
提供一个只发送4个模拟点的发布器，不需要雷达即可练习通信。
尚未实现障碍判断、距离筛选、震动强度映射、马达控制。

环境
----
已测试：Jetson aarch64、Ubuntu 20.04、ROS Noetic、Python 3.8。
依赖：rospy、std_msgs、catkin、已构建的livox_ros_driver2消息包。
不包含Livox驱动或Point-LIO源码，它们是独立依赖。
驱动参考：https://github.com/Livox-SDK/livox_ros_driver2

构建
----
本仓库本身是haptic_feedback这个catkin包，放进工作空间src下：

source /opt/ros/noetic/setup.bash
# 加载自己已构建的livox_ros_driver2工作空间环境，再构建本包。
mkdir -p ~/haptic_ws/src
cd ~/haptic_ws/src
git clone https://github.com/yewildy/wearable-slam-haptic.git haptic_feedback
chmod +x haptic_feedback/scripts/*.py
cd ~/haptic_ws
catkin_make -j2
source devel/setup.bash

在现有研究设备上，驱动环境已由mid360_pointlio_ws的环境继承。
在其他电脑上需要先安装并构建ROS 1版livox_ros_driver2，不能只克隆本包。

最小模拟演示
------------
终端A：
source ~/haptic_ws/devel/setup.bash
roslaunch haptic_feedback demo.launch

终端B：
source ~/haptic_ws/devel/setup.bash
rostopic echo -n 1 /haptic/point_count

预期输出：data: 4
模拟输入为/haptic/demo/lidar，不向真实的/livox/lidar发布数据。
测试结束在终端A按Ctrl+C。先停止模拟演示，再切换真实雷达模式。

真实输入
--------
先用自己原有的launch启动雷达驱动，确认：
rostopic type /livox/lidar

必须是livox_ros_driver2/CustomMsg，然后运行：
source ~/haptic_ws/devel/setup.bash
roslaunch haptic_feedback bringup.launch

新节点和Point-LIO可以同时订阅同一驱动话题，不修改原驱动或算法。
本次只完成模拟通信测试，没有验证真实雷达输入。
没有点数输出不能理解成“没有障碍”。

文件
----
scripts/haptic_feedback_node.py：订阅与点数发布。
scripts/demo_livox_publisher.py：4个模拟点发布器。
launch/bringup.launch：真实话题输入，支持input_topic参数。
launch/demo.launch：模拟发布器与订阅节点一起启动。
TOPIC_DEMO.txt：更详细的学习说明。
test_topic_demo.py：Jetson专用集成测试，日志路径是当前设备的路径。
verification.json：2026-10-21已执行的模拟测试结果，不代表真实雷达验证。
HISTORY.txt：开发日期与本次整理范围。

包内维护者邮箱和license仍为catkin模板占位项，尚未选择开源许可证。
本原型不能用于依赖其保障人身安全的场景。
