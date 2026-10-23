"""第三步纯逻辑：空间聚类、目标关联、相对接近速度与有限停留衰减。

全部系数是可测试的原型默认值，不是通过人体实验确定的安全阈值。
世界坐标记忆只减少旧背景的“新出现”增强，不删除当前障碍，也不证明自身身份。
"""
from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class MotionSettings:
    voxel_m: float = .06
    cluster_radius_m: float = .16
    min_cluster_points: int = 4
    max_voxels: int = 5000
    max_pairs: int = 150000
    max_tracks: int = 64
    association_m: float = .25
    ambiguity_m: float = .035
    max_speed_mps: float = 3.
    max_gap_s: float = .5
    history: int = 10
    speed_min_samples: int = 3
    speed_min_span_s: float = .2
    speed_fit_error_m: float = .03
    stop_speed_mps: float = .05
    stop_hold_s: float = 2.
    decay_tau_s: float = 2.
    stopped_floor: float = .4
    close_floor: float = .6
    speed_gain: float = .4
    speed_reference_mps: float = 1.
    ttc_gain: float = .4
    ttc_horizon_s: float = 2.
    novelty_gain: float = .15
    novelty_duration_s: float = .5
    memory_voxel_m: float = .06
    memory_ttl_s: float = 20.
    max_memory_cells: int = 10000
    pose_time_tolerance_s: float = .05
    pose_jump_m: float = .75
    pose_jump_angle_deg: float = 45.

    def __post_init__(self):
        counts=('min_cluster_points','max_voxels','max_pairs','max_tracks','history','speed_min_samples','max_memory_cells')
        for name,value in vars(self).items():
            if name in counts:
                if type(value) is not int or value<1:raise ValueError('Invalid count: '+name)
            elif type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Invalid finite positive parameter: '+name)
        if not (self.voxel_m<self.cluster_radius_m and self.speed_min_samples>=3
                and self.speed_min_samples<=self.history and self.speed_min_span_s<=self.max_gap_s
                and 0<self.stopped_floor<=self.close_floor<=1
                and self.stop_speed_mps<self.max_speed_mps and self.pose_jump_angle_deg<=180):
            raise ValueError('Inconsistent motion parameters')


def checked_pose(pose,stamp,settings):
    if pose is None:return None
    matrix=np.asarray(pose['matrix'],dtype=float)
    frame=pose['frame'];time=float(pose['stamp'])
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not isinstance(frame,str) or not frame or not math.isfinite(time)
            or abs(time-stamp)>settings.pose_time_tolerance_s
            or not np.allclose(matrix[3],[0,0,0,1],atol=1e-7)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3,:3]),1.,atol=1e-5)):
        raise ValueError('Invalid or unsynchronized sensor-to-world pose')
    return matrix,frame


def to_world(points,matrix):
    return np.asarray(points)@matrix[:3,:3].T+matrix[:3,3]


def clusters(points,settings):
    """用SciPy现有KD树及连通分量，不在Python里手写聚类搜索引擎。"""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    xyz=np.asarray(points,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('Invalid cluster coordinates')
    if not len(xyz):return []
    _,inverse=np.unique(np.floor(xyz/settings.voxel_m).astype(np.int64),axis=0,return_inverse=True)
    n=int(inverse.max())+1
    if n>settings.max_voxels:raise ValueError('CLUSTER_VOXEL_LIMIT')
    weights=np.bincount(inverse,minlength=n)
    centers=np.zeros((n,3));np.add.at(centers,inverse,xyz);centers/=weights[:,None]
    tree=cKDTree(centers)
    pair_count=int((tree.query_ball_point(centers,settings.cluster_radius_m,return_length=True).sum()-n)//2)
    if pair_count>settings.max_pairs:raise ValueError('CLUSTER_PAIR_LIMIT')
    pairs=tree.query_pairs(settings.cluster_radius_m,output_type='ndarray')
    if len(pairs):
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n))
        _,labels=connected_components(graph,directed=False)
    else:labels=np.arange(n)
    point_labels=labels[inverse];answer=[]
    for label in np.unique(point_labels):
        p=xyz[point_labels==label]
        if len(p)<settings.min_cluster_points:continue
        radii=np.linalg.norm(p,axis=1)
        answer.append(dict(center=p.mean(axis=0),distance=float(radii.min()),
                           closest=p[int(radii.argmin())],points=len(p),extent=float(np.ptp(p,axis=0).max())))
    if len(answer)>settings.max_tracks:raise ValueError('CLUSTER_TRACK_LIMIT')
    return sorted(answer,key=lambda q:q['distance'])


@dataclass
class Track:
    identity: int
    born: float
    last: float
    center: np.ndarray
    observations: deque = field(default_factory=deque)
    stop_since: object = None
    familiar: bool = False
    was_far: bool = False
    extent: float = .02


class MotionTracker:
    def __init__(self,settings=None):
        self.settings=settings or MotionSettings()
        self.tracks={};self.next_id=1;self.last_stamp=None;self.coordinate_mode=None
        self.world_memory={};self.last_pose=None

    def reset(self,clear_memory=True):
        self.tracks.clear();self.last_stamp=None;self.coordinate_mode=None;self.last_pose=None
        if clear_memory:self.world_memory.clear()

    def _speed(self,track):
        s=self.settings;rows=list(track.observations)
        if len(rows)<s.speed_min_samples or rows[-1][0]-rows[0][0]<s.speed_min_span_s-1e-8:
            return None,None,None
        times=np.array([q[0] for q in rows]);times-=times[-1]
        ranges=np.array([q[1] for q in rows]);a=np.column_stack([times,np.ones(len(times))])
        fit=np.linalg.lstsq(a,ranges,rcond=None)[0]
        error=float(np.sqrt(np.mean((a@fit-ranges)**2)))
        closing=-float(fit[0])
        if error>s.speed_fit_error_m or abs(closing)>s.max_speed_mps:
            return None,None,error
        positions=np.array([q[2] for q in rows]);slope=np.linalg.lstsq(a,positions,rcond=None)[0][0]
        return closing,float(np.linalg.norm(slope)),error

    def process(self,detections,stamp,distance_curve,pose=None,ignored_points=None):
        s=self.settings
        if type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp<=0:
            self.reset();raise ValueError('Invalid motion timestamp')
        if self.last_stamp is not None and (stamp<=self.last_stamp or stamp-self.last_stamp>s.max_gap_s):
            self.reset()
        checked=checked_pose(pose,stamp,s)
        pose_state='WORLD_COMPENSATED' if checked else 'RELATIVE_ONLY_NO_POSE'
        matrix,frame=checked if checked else (None,None)
        mode='world:'+frame if matrix is not None else 'sensor'
        if matrix is not None and self.last_pose is not None:
            previous,previous_frame=self.last_pose
            angle=math.degrees(math.acos(float(np.clip((np.trace(previous[:3,:3].T@matrix[:3,:3])-1)/2,-1,1))))
            if frame!=previous_frame or np.linalg.norm(matrix[:3,3]-previous[:3,3])>s.pose_jump_m or angle>s.pose_jump_angle_deg:
                self.reset();pose_state='POSE_JUMP_TRACKS_RESET'
        if mode!=self.coordinate_mode:
            self.tracks.clear();self.coordinate_mode=mode
        self.last_pose=(matrix.copy(),frame) if matrix is not None else None
        self.last_stamp=stamp
        self.world_memory={k:t for k,t in self.world_memory.items() if stamp-t<=s.memory_ttl_s}
        self.tracks={i:t for i,t in self.tracks.items() if stamp-t.last<=s.max_gap_s}
        positions=[to_world(d['center'][None,:],matrix)[0] if matrix is not None else d['center'] for d in detections]
        old=list(self.tracks.values());matches={};ambiguous=0
        if old and detections:
            from scipy.optimize import linear_sum_assignment
            cost=np.array([[np.linalg.norm(t.center-p) for p in positions] for t in old])
            gates=np.array([s.association_m+s.max_speed_mps*(stamp-t.last) for t in old])[:,None]
            bounded=np.where(cost<=gates,cost,1e6)
            row,col=linear_sum_assignment(bounded)
            for a,b in zip(row,col):
                if bounded[a,b]>=1e6:continue
                row_values=np.sort(bounded[a]);column_values=np.sort(bounded[:,b])
                uncertain=(len(row_values)>1 and row_values[1]-row_values[0]<s.ambiguity_m) or (len(column_values)>1 and column_values[1]-column_values[0]<s.ambiguity_m)
                if uncertain:ambiguous+=1;continue
                matches[int(b)]=old[int(a)]
        results=[];live={}
        for index,d in enumerate(detections):
            center=positions[index];track=matches.get(index)
            if track is None:
                track=Track(self.next_id,stamp,stamp,center.copy());self.next_id+=1
            track.center=center.copy();track.last=stamp
            track.was_far=track.was_far or d['distance']>.5
            track.extent=d['extent']
            if matrix is not None:
                key=tuple(np.floor(to_world(d['closest'][None,:],matrix)[0]/s.memory_voxel_m).astype(int))
                track.familiar=track.familiar or key in self.world_memory
            track.observations.append((stamp,d['distance'],center.copy()))
            while len(track.observations)>s.history:track.observations.popleft()
            closing,position_speed,error=self._speed(track)
            # 宽墙/大面积连通背景的中心随可见区域变化，不能当可靠目标运动速度。
            if d['extent']>1.:closing=position_speed=None
            stopped=(closing is not None and abs(closing)<=s.stop_speed_mps and
                     matrix is not None and position_speed<=.1)
            track.stop_since=(stamp if track.stop_since is None else track.stop_since) if stopped else None
            base=float(distance_curve(d['distance']));factor=1.
            if track.stop_since is not None:
                elapsed=max(0.,stamp-track.stop_since-s.stop_hold_s)
                factor=s.stopped_floor+(1-s.stopped_floor)*math.exp(-elapsed/s.decay_tau_s)
                if d['distance']<=.5:factor=max(factor,s.close_floor)
            # 与停止判定共用死区，不能让浮点级“接近”把衰减重新顶到base。
            closing_positive=closing if closing is not None and closing>s.stop_speed_mps else 0.
            ttc=d['distance']/closing_positive if closing_positive>s.stop_speed_mps else None
            motion=s.speed_gain*min(1.,closing_positive/s.speed_reference_mps)
            if ttc is not None:motion+=s.ttc_gain*max(0.,1-ttc/s.ttc_horizon_s)
            novel=not track.familiar and stamp-track.born<s.novelty_duration_s and d['distance']<1.2
            novelty=s.novelty_gain if novel else 0.
            value=min(1.,max(base*factor,base+motion+novelty if motion or novelty else 0.))
            results.append(dict(track_id=track.identity,distance_m=d['distance'],closing_speed_mps=closing,
                                coordinate_speed_mps=position_speed,ttc_s=ttc,fit_error_m=error,
                                strength=value,base_strength=base,dwell_factor=factor,
                                familiar_background=track.familiar,novelty_boost=novelty,
                                points=d['points'],wide_surface_velocity_rejected=d['extent']>1.))
            live[track.identity]=track
        # 短暂遮挡不把同一目标立即改成新ID；未观测轨迹不会进入当前强度输出。
        pending={i:t for i,t in self.tracks.items() if i not in live and stamp-t.last<=s.max_gap_s}
        slots=max(0,s.max_tracks-len(live))
        recent=sorted(pending,key=lambda i:pending[i].last,reverse=True)[:slots]
        self.tracks=dict(live)
        self.tracks.update({i:pending[i] for i in recent})
        # 本帧被忽略的点只记到以后，不用于把本帧新目标立即说成旧背景。
        if matrix is not None and ignored_points is not None and len(ignored_points):
            keys=np.unique(np.floor(to_world(ignored_points,matrix)/s.memory_voxel_m).astype(int),axis=0)
            for key in keys:self.world_memory[tuple(key)]=stamp
            if len(self.world_memory)>s.max_memory_cells:
                keep=sorted(self.world_memory,key=self.world_memory.get,reverse=True)[:s.max_memory_cells]
                self.world_memory={k:self.world_memory[k] for k in keep}
        return dict(tracks=results,pose_state=pose_state,association_uncertain_count=ambiguous,
                    background_memory_cells=len(self.world_memory),
                    speed_kind='relative_radial_closing_not_object_approach_identity',
                    physical_validation=False)
