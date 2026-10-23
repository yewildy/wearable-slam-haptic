"""滚动过滤后独立去噪；邻域支持不是自身身份判断。"""
import math
import numpy as np
from distance_policy import DistancePolicy, supported_points


class SupportedDistancePolicy(DistancePolicy):
    def support_mask(self, xyz):
        # 已通过一次邻域去噪和剩余组大小检查，不递归侵蚀物体边缘。
        return np.ones(len(xyz), dtype=bool)

    def _support(self, xyz):
        if self.backend != 'scipy' or not len(xyz):
            return supported_points(xyz, self.settings.support_radius,
                                    self.settings.min_neighbors, self.backend)
        from scipy.spatial import cKDTree
        # 只需知道第4近点是否在半径内，不枚举稠密区域的全部邻居。
        distances, _ = cKDTree(xyz).query(xyz, k=self.settings.min_neighbors+1)
        return distances[:, -1] <= self.settings.support_radius

    def _group_mask(self, xyz):
        """保留去噪后仍有至少4个点的连通组，连接半径沿用8cm。

        不是再次要求每个点有3个邻居：链状组的边缘也能保留。
        只删除整个小组，一次完成，不重复剥掉边缘直到收敛。
        """
        count = len(xyz)
        minimum = self.settings.min_neighbors + 1
        if count < minimum:
            return np.zeros(count, dtype=bool)
        radius = self.settings.support_radius
        if self.backend == 'scipy':
            from scipy.spatial import cKDTree
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            # 每点最多保存minimum条边，避免稠密云产生全部点对。
            # 少于minimum点的小组件不可能因截边而产生：若某点截掉了
            # 邻居，它已经连接minimum-1个其他点，所在组件必定足够大。
            distances, neighbors = cKDTree(xyz).query(xyz, k=minimum)
            rows = np.broadcast_to(np.arange(count)[:, None], neighbors.shape)
            within = (distances <= radius) & (neighbors < count)
            graph = coo_matrix((np.ones(np.count_nonzero(within)),
                                (rows[within], neighbors[within])), shape=(count, count))
            _, labels = connected_components(graph.tocsr(), directed=False)
            return np.bincount(labels)[labels] >= minimum
        if self.backend != 'reference' or count > 4000:
            raise ValueError('Reference group check requires at most 4000 test points')
        # 教学/测试参考版遍历完整半径图，用来核对设备的稀疏图实现。
        unseen = np.ones(count, dtype=bool)
        keep = np.zeros(count, dtype=bool)
        for seed in range(count):
            if not unseen[seed]:
                continue
            component = [seed]
            unseen[seed] = False
            for index in component:
                delta = xyz - xyz[index]
                adjacent = np.flatnonzero(unseen & (np.sum(delta*delta, axis=1) <= radius*radius))
                unseen[adjacent] = False
                component.extend(adjacent.tolist())
            if len(component) >= minimum:
                keep[component] = True
        return keep

    def process(self, points, frame, stamp, source_age_s=0., declared_count=None, now=None):
        xyz = np.asarray(points)
        self.filtered_points = np.empty((0, 3))
        valid = (xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype.kind in 'fi'
                 and frame == self.settings.expected_frame
                 and type(stamp) in (int, float) and math.isfinite(stamp) and stamp > 0
                 and math.isfinite(source_age_s)
                 and -self.settings.future_tolerance <= source_age_s <= self.settings.stale_timeout
                 and (self.last_stamp is None or stamp > self.last_stamp)
                 and (declared_count is None or type(declared_count) is int and declared_count == len(xyz)))
        if not valid:
            return super().process(points, frame, stamp, source_age_s, declared_count, now)
        finite = np.isfinite(xyz).all(axis=1)
        ranges = np.linalg.norm(xyz, axis=1)
        eligible = finite & (ranges > 0) & (ranges <= self.settings.max_range)
        active = np.flatnonzero(eligible)
        # 单次半径离群点过滤。邻域支持来自本帧原始剩余点，不作递归k-core。
        active = active[self._support(xyz[active])]
        grouped = self._group_mask(xyz[active])
        small_group_removed = int(np.count_nonzero(~grouped))
        active = active[grouped]
        keep = ~eligible
        keep[active] = True
        self.filtered_points = xyz[keep]
        report = super().process(self.filtered_points, frame, stamp, source_age_s, now=now)
        report.update(isolated_removed_points=int(np.count_nonzero(~keep)),
                      postfilter_small_group_removed_points=small_group_removed,
                      isolation_input_points=len(xyz),
                      isolation_policy='single_pass_radius_then_connected_group',
                      isolation_group_min_points=self.settings.min_neighbors+1,
                      isolation_support_radius_m=self.settings.support_radius,
                      thin_object_miss_risk=True)
        return report
