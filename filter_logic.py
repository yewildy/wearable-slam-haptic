"""第一阶段的学习代码：过滤雷达扫描到的近点。

主要编辑 PointFilter.filter()。这个文件不依赖 ROS，可在电脑上单独运行。
初始版本原样返回有效点，还没有实现自身排除或滚动统计。
"""
import numpy as np


class PointFilter:

  def __init__(self):
    # 一、准备参数：50厘米也就是 0.50 米
    self.near_radius_m = 0.50
    self.window_seconds = 10.0

  def filter(self, points, stamp):
    """二、在这里编写自己的规则（初学者易懂的循环版）。"""
    # 创建一个空列表，用来装筛选后保留下来的点
    kept_points = []

    # 1. 用循环一个一个地检查点
    for p in points:
      x = p[0]  # 取出当前点的 x 坐标
      y = p[1]  # 取出当前点的 y 坐标
      z = p[2]  # 取出当前点的 z 坐标

      # 2. 计算到雷达原点的距离：平方根(x^2 + y^2 + z^2)
      # 在 Python 里，开平方可以用 ** 0.5（也就是开二分之一次方）
      distance = (x**2 + y**2 + z**2) ** 0.5

      # 3. 如果距离大于 0.5 米，说明不是近点，把它收集起来
      if distance > self.near_radius_m:
        kept_points.append(p)

    # 4. 把收集好的普通列表，重新转回 NumPy 数组并返回
    return np.array(kept_points)


if __name__ == '__main__':
    # 三、电脑上的小例子。不连接雷达，不启动 ROS。
    example = np.array([
        [0.10, 0.00, 0.00],
        [0.30, 0.00, 0.00],
        [0.80, 0.00, 0.00],
        [1.20, 0.00, 0.00],
    ], dtype=float)
    point_filter = PointFilter()
    result = point_filter.filter(example, stamp=1.0)
    print('输入点：\n', example)
    print('保留点：\n', result)
    print('当前是原样返回；近点排除尚未实现。')
