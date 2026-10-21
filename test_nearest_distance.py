import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parent / 'scripts' / 'haptic_feedback_node.py'
if not path.exists():
    path = Path('/home/eno/haptic_ws/src/haptic_feedback/scripts/haptic_feedback_node.py')
spec = importlib.util.spec_from_file_location('haptic_distance_node', path)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


def point(x, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


class DistanceTests(unittest.TestCase):
    def test_four_points(self):
        self.assertEqual(node.nearest_distance([point(x) for x in (1, 2, 3, 4)]), 1.0)

    def test_not_just_x(self):
        self.assertEqual(node.nearest_distance([point(3, 4), point(6)]), 5.0)

    def test_three_dimensions(self):
        self.assertEqual(node.nearest_distance([point(1, 2, 2)]), 3.0)

    def test_negative_coordinate(self):
        self.assertEqual(node.nearest_distance([point(-2)]), 2.0)

    def test_empty_is_unknown(self):
        self.assertTrue(math.isnan(node.nearest_distance([])))

    def test_nonfinite_values(self):
        self.assertEqual(node.nearest_distance([point(float('nan')), point(float('inf')), point(2)]), 2.0)
        self.assertTrue(math.isnan(node.nearest_distance([point(float('nan'))])))

    def test_origin_is_math_zero(self):
        self.assertEqual(node.nearest_distance([point(0)]), 0.0)


if __name__ == '__main__':
    unittest.main()
