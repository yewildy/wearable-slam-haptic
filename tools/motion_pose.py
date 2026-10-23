"""只读现有TF与固定Point-LIO外参；不发布TF，不猜测缺失的坐标变换。"""
import numpy as np
from motion_tracking import checked_pose,MotionSettings


def fixed_extrinsic(rotation,translation,estimated=False):
    if estimated is not False:raise ValueError('Online estimated extrinsic cannot use a static snapshot')
    r=np.asarray(rotation,dtype=float).reshape(3,3)
    t=np.asarray(translation,dtype=float)
    if t.shape!=(3,):raise ValueError('Invalid extrinsic translation')
    result=np.eye(4);result[:3,:3]=r;result[:3,3]=t
    checked_pose(dict(matrix=result,frame='body',stamp=1.),1.,MotionSettings())
    return result


class PoseProvider:
    def __init__(self,rospy,tf):
        self.rospy,self.tf=rospy,tf
        self.listener=tf.TransformListener()
        self.world=rospy.get_param('~world_frame','camera_init')
        self.body=rospy.get_param('~body_frame','body')
        self.prefix=rospy.get_param('~pointlio_parameter_prefix','/laserMapping/mapping')

    def at(self,frame,stamp):
        try:
            tr,q=self.listener.lookupTransform(self.world,frame,stamp)
            result=np.asarray(self.listener.fromTranslationRotation(tr,q),dtype=float)
            return dict(matrix=result,frame=self.world,stamp=stamp.to_sec()),'DIRECT_TF'
        except (self.tf.LookupException,self.tf.ConnectivityException,self.tf.ExtrapolationException):
            pass
        if frame!='livox_frame':return None,'NO_VERIFIED_SENSOR_TRANSFORM'
        try:
            # 每次读取固定外参声明，运行中参数改变也不能继续使用旧快照。
            estimate=self.rospy.get_param(self.prefix+'/extrinsic_est_en')
            extrinsic=fixed_extrinsic(self.rospy.get_param(self.prefix+'/extrinsic_R'),
                                      self.rospy.get_param(self.prefix+'/extrinsic_T'),estimate)
            tr,q=self.listener.lookupTransform(self.world,self.body,stamp)
            world_body=np.asarray(self.listener.fromTranslationRotation(tr,q),dtype=float)
            result=world_body@extrinsic
            return dict(matrix=result,frame=self.world,stamp=stamp.to_sec()),'EXISTING_POINTLIO_FIXED_EXTRINSIC'
        except (KeyError,ValueError,TypeError,self.tf.LookupException,self.tf.ConnectivityException,self.tf.ExtrapolationException):
            return None,'POSE_UNAVAILABLE_NO_IDENTITY_ASSUMPTION'
