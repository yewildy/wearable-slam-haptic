"""A synthetic counterexample: radius pruning can leave a lone supported center."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.code))
    from motion_pipeline import MotionEngine
    engine=MotionEngine()
    far=np.array([[.8,0,0],[.8,.01,0],[.8,0,.01],[.8,.01,.01]])
    for i in range(110):
        engine.process(far,'livox_frame',1.+i*.1)
    # 中心有三个8cm内邻居；三个外围点彼此不够近，各自只有中心一个邻居。
    star=np.array([[.1,0,0],[.1,.07,0],[.1,-.07,0],[.1,0,.07]])
    result=engine.process(np.vstack([far,star]),'livox_frame',12.)
    kept=engine.base.filtered_points
    surviving_near=int(np.count_nonzero(np.linalg.norm(kept,axis=1)<.5))
    answer=dict(kind='synthetic_counterexample_not_physical_identity',
                star_input_points=4,isolated_removed_points=result.get('isolated_removed_points'),
                surviving_near_points=surviving_near,state=result['state'],
                distance_m=result['distance_m'],strength=result['strength'],
                nearby_tracks=[t['points'] for t in result['tracks'] if t['distance_m']<.5],
                violates_grouped_target_requirement=bool(surviving_near==1 and result['valid']
                    and result['distance_m']<.5),production_code_changed=False)
    assert answer['violates_grouped_target_requirement'],answer
    args.output.write_text(json.dumps(answer,indent=2,allow_nan=False))
    print(json.dumps(answer,indent=2))


if __name__=='__main__':main()
