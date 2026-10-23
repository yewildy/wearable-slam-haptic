"""第一/二步保持原规则；第三步只在它们给出可用结果后融合运动信息。"""
import numpy as np
from rolling_near import RollingNear
from rolling_distance import rolling_result
from distance_policy import DistancePolicy,Settings,distance_strength
from motion_tracking import MotionTracker,MotionSettings,clusters,checked_pose,to_world
from motion_background import BackgroundContinuity
from motion_support import SupportedDistancePolicy


class CapturedRolling(RollingNear):
    def __init__(self):
        super().__init__();self.removed=None;self.guard=None;self.adjuster=None

    def reset(self):
        super().reset();self.removed=None

    def process(self,*args,**kwargs):
        removed,detail=super().process(*args,**kwargs)
        original_count=int(removed.sum())
        protected=0
        xyz=np.asarray(args[0])
        near=np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)>0)&(np.linalg.norm(xyz,axis=1)<=.5)
        allowed=self.guard(xyz,near) if self.guard is not None else np.zeros(len(xyz),dtype=bool)
        protected=int(np.count_nonzero(removed&allowed))
        removed=removed&~allowed
        if self.adjuster is not None and detail['rolling_ready']:
            removed,extra=self.adjuster(xyz,removed);detail.update(extra)
            protected+=int(np.count_nonzero(removed&allowed))
            removed=removed&~allowed
        detail.update(rolling_raw_removed_points=original_count,
                      rolling_removed_points=int(removed.sum()),rolling_protected_track_points=protected)
        self.removed=removed.copy()
        return removed,detail


class MotionEngine:
    def __init__(self,settings=None,backend='scipy'):
        self.motion=MotionTracker(settings or MotionSettings())
        self.base=SupportedDistancePolicy(Settings(),backend=backend)
        self.rolling=CapturedRolling()
        self.background=BackgroundContinuity()

    def _track_guard(self,pose,stamp):
        try:checked=checked_pose(pose,stamp,self.motion.settings)
        except (ValueError,KeyError,TypeError):checked=None
        if checked is None:return None
        matrix,frame=checked
        if self.motion.coordinate_mode!='world:'+frame:return None
        prior=[];radii=[]
        for t in self.motion.tracks.values():
            speed,_,_=self.motion._speed(t)
            if t.was_far and speed is not None and t.extent<=.4 and stamp-t.last<=.5:
                prior.append(t.center);radii.append(min(.08,max(.025,.866*t.extent+.015)))
        if not prior:return None
        def guard(xyz,removed):
            from scipy.spatial import cKDTree
            protected=np.zeros(len(xyz),dtype=bool)
            indices=np.flatnonzero(removed)
            distance,nearest=cKDTree(np.asarray(prior)).query(to_world(xyz[indices],matrix),k=1)
            protected[indices]=distance<=np.asarray(radii)[nearest]
            return protected
        return guard

    def reset(self):
        self.motion.reset();self.rolling.reset();self.background.reset()

    def unknown(self,reason):
        self.reset()
        return dict(self.base.unknown(reason),algorithm_mode='motion_v3',
                    motion_valid=False,pose_state='UNAVAILABLE',tracks=[],motor_control=False)

    def process(self,points,frame,stamp,age=0.,count=None,pose=None,now=None):
        xyz=np.asarray(points)
        count=len(xyz) if count is None else count
        if (type(stamp) in (int,float) and np.isfinite(stamp)
                and self.base.last_stamp is not None and stamp<=self.base.last_stamp):
            # 退时的这一帧未知；之后从新的10秒历史重新开始，不沿用旧速度。
            self.reset();self.base.last_stamp=None
            return self.unknown('TIMESTAMP_NONINCREASING')
        if pose is None and (self.motion.coordinate_mode or '').startswith('world:'):
            # 丢位姿时不借旧世界记忆继续屏蔽；重新统计并明确未知。
            self.reset()
        self.rolling.guard=self._track_guard(pose,stamp)
        try:checked=checked_pose(pose,stamp,self.motion.settings)
        except (ValueError,TypeError,KeyError):checked=None
        self.rolling.adjuster=lambda xyz,removed:self.background.apply(xyz,removed,checked,stamp)
        base=rolling_result(self.base,self.rolling,xyz,frame,stamp,age,count,now)
        base.update(algorithm_mode='motion_v3',motion_valid=False,
                    pose_state='UNAVAILABLE',tracks=[],coefficients='prototype_not_empirically_fitted')
        if not base['valid']:
            # 未知输入不能借上一帧速度变成有效强度。
            if base['state'] in ('NO_REMAINING_OBSERVATIONS','NO_SUPPORTED_POINTS','UNSUPPORTED_CLOSER_RETURN'):
                try:self.motion.process([],stamp,lambda d:distance_strength(d,self.base.settings),pose)
                except (ValueError,TypeError,KeyError):self.motion.reset()
            else:self.motion.reset()
            return base
        if self.rolling.removed is None:
            return self.unknown('ROLLING_PARTITION_UNAVAILABLE')
        cleaned=self.base.filtered_points
        finite=np.isfinite(cleaned).all(axis=1)
        radii=np.linalg.norm(cleaned,axis=1)
        keep=finite&(radii>=self.base.settings.min_range)&(radii<=self.base.settings.max_range)
        try:
            detections=clusters(cleaned[keep],self.motion.settings)
            extra=self.motion.process(detections,stamp,lambda d:distance_strength(d,self.base.settings),
                                      pose,xyz[self.rolling.removed])
        except (ValueError,TypeError,KeyError) as error:
            self.motion.reset()
            base.update(motion_error=str(error),pose_state='MOTION_REJECTED_DISTANCE_FALLBACK')
            return base
        base.update(extra)
        base['nearest_distance_m']=base['distance_m']
        base['distance_only_strength']=base['strength']
        if base['state']=='NEAR_FIELD_ALERT':
            # 不让第三步把尚未处理好的10cm近点通过“习惯了”衰减掉。
            base.update(motion_override='UNTRUSTED_NEAR_FIELD_NOT_DECAYED')
            return base
        if not extra['tracks']:
            base.update(motion_override='NO_TRACK_DISTANCE_FALLBACK')
            return base
        selected=max(extra['tracks'],key=lambda t:t['strength'])
        # 最近的有效点若未形成轨迹，仍保留第二步强度，不能被远处轨迹替代掉。
        covered=base['distance_m'] is not None and any(t['distance_m']<=base['distance_m']+.01 for t in extra['tracks'])
        strength=selected['strength'] if covered else max(base['strength'],selected['strength'])
        if not covered and base['strength']>selected['strength']:
            base.update(motion_override='UNTRACKED_NEAREST_DISTANCE_FLOOR');return base
        base.update(motion_valid=True,strength=strength,distance_m=selected['distance_m'],
                    base_state=base['state'],distance_level=base['level'],
                    state='DISTANCE_CANDIDATE' if strength>0 else 'NO_NEAR_CANDIDATE',
                    level=3 if strength>=.75 else 2 if strength>.25 else 1 if strength>0 else 0,
                    selected_track_id=selected['track_id'],closing_speed_mps=selected['closing_speed_mps'],
                    ttc_s=selected['ttc_s'],dwell_factor=selected['dwell_factor'],
                    familiar_background=selected['familiar_background'])
        return base
