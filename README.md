# wearable-slam-haptic

## English

This project grew out of my wearable SLAM work, which I resumed in September 2026. I am using a Jetson and a MID-360. In October, I started working through the Python point-cloud processing code myself.

So far, the node receives `/livox/lidar`, removes points within 50 cm of the LiDAR, and publishes the remaining cloud on `/haptic_learning/filtered_points`. It also calculates the nearest remaining distance and prints it in the terminal. The distance does not have its own output topic yet.

The current calculation is in `filter_logic_closer_num.py`. `lidar_node.py` handles ROS messages. `filter_logic.py` is the earlier version, and `mock_lidar.py` sends four points per second for practice.

I want to try a simple linear mapping next: a shorter distance gives a stronger alert. The motor is not connected yet. Removing every point within 50 cm is only a first exercise, not a way to identify the wearer's body.

The code uses ROS Noetic and the built `livox_ros_driver2` message package. `run.sh` loads the workspace on my current Jetson. With the LiDAR driver running:

```bash
bash run.sh _input_topic:=/livox/lidar
```

For the simulated input, start `mock_lidar.py` in a ROS environment with a running master, then use:

```bash
bash run.sh _input_topic:=/haptic_learning/mock_lidar
```

## 中文

这个项目是在2026年9月恢复的可穿戴SLAM研究基础上逐步开展的，设备使用Jetson和MID-360。10月开始自己逐步编写和理解Python点云处理代码。

目前节点可以接收 `/livox/lidar`，删除距离雷达50 cm以内的点，再通过 `/haptic_learning/filtered_points` 发布剩余点云。也能计算剩余点中的最近距离，并显示在终端。距离还没有单独的输出话题。

当前计算写在 `filter_logic_closer_num.py`，ROS收发在 `lidar_node.py`。`filter_logic.py` 保留了之前的版本，`mock_lidar.py` 每秒发送四个点用于练习。

下一步想先试简单的线性映射：距离越近，提醒越强。暂时不接马达。当前直接删除50 cm内所有点只是初步练习，还不能识别哪些点属于佩戴者身体。

运行环境是ROS Noetic和已构建的 `livox_ros_driver2` 消息包。上面的两条命令分别用于真实和模拟输入，`run.sh` 加载的是目前Jetson上的工作空间。
