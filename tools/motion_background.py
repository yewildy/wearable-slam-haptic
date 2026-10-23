"""小幅自身移动时延续已忽略区域；不学习新近点，不放宽成整个半径屏蔽。"""
import math
import numpy as np
from motion_tracking import to_world


class BackgroundContinuity:
    def __init__(self):
        self.cells={};self.frame=None

    def reset(self):
        self.cells.clear();self.frame=None

    def apply(self,xyz,raw_removed,checked,stamp):
        raw_removed=np.asarray(raw_removed,dtype=bool)
        output=raw_removed.copy();extra=0
        if checked is None:
            self.reset();return output,dict(rolling_continuity_removed_points=0)
        matrix,frame=checked
        if self.frame!=frame:self.reset();self.frame=frame
        self.cells={k:v for k,v in self.cells.items() if stamp-v['seen']<=10.}
        ranges=np.linalg.norm(xyz,axis=1)
        near=np.isfinite(xyz).all(axis=1)&(ranges>0)&(ranges<=.5)
        indices=np.flatnonzero(near&~raw_removed)
        records=list(self.cells.values())
        if records and len(indices):
            from scipy.spatial import cKDTree
            world=to_world(xyz[indices],matrix)
            distance,nearest=cKDTree(np.array([r['point'] for r in records])).query(world,k=1)
            for i,d,match in zip(indices,distance,nearest):
                if d>.015:continue
                record=records[int(match)]
                delta=np.linalg.norm(matrix[:3,3]-record['pose'][:3,3])
                relative=record['pose'][:3,:3].T@matrix[:3,:3]
                angle=math.degrees(math.acos(float(np.clip((np.trace(relative)-1)/2,-1,1))))
                if (stamp-record['anchor']<=10. and delta<=.05 and angle<=5.
                        and ranges[i]>=record['range']-.03):
                    output[i]=True;extra+=1
        # 只用原始滚动判断已接受的点刷新记忆，禁止用本次延续出的点滚雪球。
        raw_indices=np.flatnonzero(near&raw_removed)
        if len(raw_indices):
            world=to_world(xyz[raw_indices],matrix)
            keys=np.floor(world/.02).astype(np.int64)
            for i,point,key in zip(raw_indices,world,keys):
                key=tuple(key);old=self.cells.get(key)
                if old is None or stamp-old['anchor']>10.:
                    self.cells[key]=dict(point=point.copy(),range=float(ranges[i]),pose=matrix.copy(),
                                         anchor=stamp,seen=stamp)
                else:old['seen']=stamp
            if len(self.cells)>10000:
                keep=sorted(self.cells,key=lambda k:self.cells[k]['seen'],reverse=True)[:10000]
                self.cells={k:self.cells[k] for k in keep}
        return output,dict(rolling_continuity_removed_points=extra,
                           continuity_rule='same_world_point_small_ego_move_only',
                           continuity_memory_cells=len(self.cells))
