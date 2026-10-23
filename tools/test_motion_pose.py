import types,unittest
import numpy as np
from motion_pose import PoseProvider


class Missing(Exception):pass


class PoseTests(unittest.TestCase):
    def make(self,direct=False,missing=False,estimated=False):
        parameters={'/laserMapping/mapping/extrinsic_R':list(np.eye(3).ravel()),
                    '/laserMapping/mapping/extrinsic_T':[.1,.2,.3],
                    '/laserMapping/mapping/extrinsic_est_en':estimated}
        def param(name,*default):
            if name in parameters and not missing:return parameters[name]
            if default:return default[0]
            raise KeyError(name)
        def lookup(target,source,stamp):
            if source=='livox_frame' and not direct:raise Missing()
            return [1.,2.,3.],[0,0,0,1]
        def matrix(t,q):
            m=np.eye(4);m[:3,3]=t;return m
        listener=types.SimpleNamespace(lookupTransform=lookup,fromTranslationRotation=matrix)
        tf=types.SimpleNamespace(TransformListener=lambda:listener,LookupException=Missing,
                                 ConnectivityException=Missing,ExtrapolationException=Missing)
        return PoseProvider(types.SimpleNamespace(get_param=param),tf)

    def test_existing_static_extrinsic_used_without_publishing_tf(self):
        result,source=self.make().at('livox_frame',types.SimpleNamespace(to_sec=lambda:5.))
        np.testing.assert_allclose(result['matrix'][:3,3],[1.1,2.2,3.3])
        self.assertEqual(source,'EXISTING_POINTLIO_FIXED_EXTRINSIC')

    def test_direct_sensor_tf_is_preferred(self):
        result,source=self.make(direct=True).at('livox_frame',types.SimpleNamespace(to_sec=lambda:5.))
        np.testing.assert_allclose(result['matrix'][:3,3],[1.,2.,3.])
        self.assertEqual(source,'DIRECT_TF')

    def test_missing_or_online_extrinsic_never_falls_back_to_identity(self):
        for provider in (self.make(missing=True),self.make(estimated=True)):
            result,source=provider.at('livox_frame',types.SimpleNamespace(to_sec=lambda:5.))
            self.assertIsNone(result);self.assertEqual(source,'POSE_UNAVAILABLE_NO_IDENTITY_ASSUMPTION')


if __name__=='__main__':unittest.main()
