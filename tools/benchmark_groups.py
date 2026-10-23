"""Bounded synthetic CPU comparison, not live timing or physical accuracy."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code))
    from motion_support import SupportedDistancePolicy
    spec = importlib.util.spec_from_file_location('baseline_support', args.baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    results = {}
    rng = np.random.default_rng(24)
    for count in (2000, 20000):
        points = rng.uniform(-.2, .2, (count, 3)) + [1., 0., 0.]
        results[str(count)] = {}
        for label, policy_type in [('before', old.SupportedDistancePolicy),
                                   ('after', SupportedDistancePolicy)]:
            policy = policy_type()
            elapsed = []
            for i in range(6):
                start = time.perf_counter()
                report = policy.process(points, 'livox_frame', 1. + i*.1)
                duration = time.perf_counter()-start
                if i:
                    elapsed.append(duration)
            assert report['valid'] and report['isolated_removed_points'] == 0
            results[str(count)][label] = dict(median_s=float(np.median(elapsed)),
                max_s=max(elapsed), repetitions=len(elapsed), kept=len(policy.filtered_points))
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
