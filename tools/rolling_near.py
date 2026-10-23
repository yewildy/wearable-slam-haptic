"""滚动稳定近点实验；稳定只表示时间特征，不证明自身身份。"""
from collections import deque, Counter
import numpy as np


class RollingNear:
    def __init__(self):
        self.frames = deque()
        self.last_stamp = None
        self.started_stamp = None

    def reset(self):
        self.frames.clear()
        self.last_stamp = None
        self.started_stamp = None

    def process(self, xyz, stamp, frame='livox_frame'):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(stamp):
            self.reset()
            raise ValueError('Invalid cloud or timestamp')
        if frame != 'livox_frame':
            self.reset()
            raise ValueError('Expected sensor-local frame')
        if self.last_stamp is not None and (stamp <= self.last_stamp or stamp-self.last_stamp > .5):
            self.reset()
        self.last_stamp = stamp
        if self.started_stamp is None:
            self.started_stamp = stamp
        finite = np.isfinite(xyz).all(axis=1)
        ranges = np.linalg.norm(xyz, axis=1)
        near = finite & (ranges > 0) & (ranges <= .5)
        keys = np.zeros((len(xyz), 3), dtype=np.int64)
        keys[near] = np.floor(xyz[near]/.02).astype(np.int64)
        cells = set(map(tuple, keys[near]))
        # 先用过去的数据判断当前帧，当前点不会在第一次出现时屏蔽自己。
        while self.frames and self.frames[0][0] < stamp-10.-1e-6:
            self.frames.popleft()
        # 是否完成启动等待不能取决于最老保留帧的位置，否则低于10Hz时每次裁窗会来回跳。
        ready = (len(self.frames) >= 30 and stamp-self.started_stamp >= 10.-1e-6
                 and stamp-self.frames[0][0] >= 9.5)
        accepted = set()
        if ready:
            # 非重复扫描不保证每帧命中同一格；每0.5秒合并一次命中，仍只使用过去10秒。
            buckets = [set() for _ in range(20)]
            for t, row in self.frames:
                index = min(19, max(0, int((t-(stamp-10.))/.5)))
                buckets[index].update(row)
            counts = Counter(k for row in buckets for k in row)
            accepted = {k for k,n in counts.items() if n >= 16}
            # 五个2秒子窗口都要有支持，不能只在窗口的一端出现。
            for segment in range(5):
                rows = buckets[segment*4:(segment+1)*4]
                support = Counter(k for row in rows for k in row)
                accepted &= {k for k,n in support.items() if n >= 2}
        removed = np.zeros(len(xyz), dtype=bool)
        indices = np.flatnonzero(near)
        removed[indices] = [tuple(keys[i]) in accepted for i in indices]
        # 量测落在2cm格边界两侧时，允许距离已稳定小格的表面最多1cm误差。
        # 不扩成整个方向或50cm球；同一容差内的新外物也可能被忽略，是已知限制。
        if accepted and len(indices):
            centers=(np.asarray(sorted(accepted),dtype=float)+.5)*.02
            query=xyz[indices]
            try:
                from scipy.spatial import cKDTree
                neighbors=cKDTree(centers).query_ball_point(query, .01+np.sqrt(3)*.01+1e-9)
            except ImportError:
                if len(query)*len(centers)>1000000:
                    raise RuntimeError('SciPy required for large rolling spatial match')
                neighbors=[np.flatnonzero(np.linalg.norm(centers-p,axis=1)<=.01+np.sqrt(3)*.01+1e-9) for p in query]
            for index, row in zip(indices,neighbors):
                if len(row) and not removed[index]:
                    delta=np.maximum(np.abs(centers[row]-xyz[index])-.01,0.)
                    removed[index]=bool(np.any(np.linalg.norm(delta,axis=1)<=.01+1e-9))
        self.frames.append((stamp, cells))
        return removed, dict(rolling_ready=ready, rolling_window_s=10., rolling_radius_m=.5,
                             rolling_history_s=stamp-self.frames[0][0],
                             rolling_cells=len(accepted), rolling_removed_points=int(removed.sum()),
                             rolling_spatial_tolerance_m=.01,
                             rolling_identity='unconfirmed_temporal_stability_only')
