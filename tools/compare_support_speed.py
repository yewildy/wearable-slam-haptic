"""Pure offline output-equivalence and CPU comparison; no ROS nodes or hardware."""
import argparse
import hashlib
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
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code))
    from motion_support import SupportedDistancePolicy
    spec = importlib.util.spec_from_file_location('previous_support', args.baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    types = [old.SupportedDistancePolicy, SupportedDistancePolicy]
    answer = dict(scope='offline_filter_only_not_whole_node', output_equivalence=True)

    def compare_frames(frames, repeat=1):
        policies = [kind() for kind in types]
        times = [[], []]
        counts = []
        for frame_index, points in enumerate(frames):
            counts.append(len(points))
            for repetition in range(repeat):
                reports = [None, None]
                # 交替测量顺序，避免总是把后测的一版放在热缓存条件下。
                for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
                    start = time.perf_counter()
                    reports[index] = policies[index].process(points, 'livox_frame',
                        1. + frame_index*repeat + repetition, now=1.)
                    elapsed = time.perf_counter()-start
                    if repeat == 1 or repetition > 0:
                        times[index].append(elapsed)
                assert reports[0] == reports[1], 'Filter reports differ'
                np.testing.assert_array_equal(policies[0].filtered_points, policies[1].filtered_points)
        return dict(frames=len(counts), min_points=min(counts), max_points=max(counts),
            timings={name: dict(median_s=float(np.median(values)), p95_s=float(np.percentile(values, 95)),
                max_s=max(values), observations=len(values)) for name, values in zip(('before', 'after'), times)})

    rng = np.random.default_rng(24)
    answer['synthetic'] = {}
    for name, count in (('dense_2000', 2000), ('dense_20000', 20000), ('dense_20000_with_isolated', 20000)):
        xyz = rng.uniform(-.2, .2, (count, 3))+[1., 0, 0]
        if name.endswith('isolated'):
            xyz = np.vstack([xyz, [[.1, 0, 0]]])
        answer['synthetic'][name] = compare_frames([xyz], repeat=6)

    before_hash = hashlib.sha256(args.sample.read_bytes()).hexdigest()
    with np.load(args.sample, allow_pickle=False) as sample:
        frames = [sample['xyz'][a:b] for a, b in zip(sample['offsets'][:-1], sample['offsets'][1:])]
        answer['saved_near_crop'] = compare_frames(frames)
    assert hashlib.sha256(args.sample.read_bytes()).hexdigest() == before_hash

    import rosbag
    with rosbag.Bag(str(args.bag), 'r') as bag:
        count = bag.get_message_count(topic_filters=['/livox/lidar'])
        indices = set(np.linspace(0, count-1, min(12, count), dtype=int).tolist())
        frames = []
        for index, (_, message, _) in enumerate(bag.read_messages(topics=['/livox/lidar'])):
            if index in indices:
                assert message.header.frame_id == 'livox_frame'
                xyz = np.array([(p.x, p.y, p.z) for p in message.points], dtype=np.float64)
                assert len(xyz) == message.point_num
                frames.append(xyz)
        assert len(frames) == len(indices) and frames
    answer['saved_full_frames'] = compare_frames(frames, repeat=2)
    answer['source_files_modified'] = False
    answer['live_launcher_changed'] = False
    args.output.write_text(json.dumps(answer, indent=2), encoding='utf-8')
    print(json.dumps(answer, indent=2))


if __name__ == '__main__':
    main()
