"""Geometry/regression checks without importing a simulator or loading a policy."""

import math
import unittest

from evaluate_walking import StraightnessSummary, projected_horizontal_forward, summarize_straightness


def yaw_quaternion(degrees):
    angle = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(angle), math.cos(angle))


class StraightnessGeometryTests(unittest.TestCase):
    def test_fixed_initial_axes_and_translation_invariance(self):
        for yaw in (0, 45, 90, -135):
            angle = math.radians(yaw)
            forward = (math.cos(angle), math.sin(angle))
            left = (-forward[1], forward[0])
            for origin in ((0, 0, 0.4), (1000, -400, 0.8)):
                path = StraightnessSummary(origin, yaw_quaternion(yaw))
                destination = (origin[0] + 2 * forward[0] + 0.3 * left[0],
                               origin[1] + 2 * forward[1] + 0.3 * left[1], origin[2])
                point = path.add(destination, yaw_quaternion(yaw + 30))
                self.assertAlmostEqual(point["along_track_progress_m"], 2)
                self.assertAlmostEqual(point["lateral_deviation_m"], 0.3)
                self.assertAlmostEqual(point["heading_drift_wrapped_rad"], math.pi / 6)

    def test_maximum_excursion_is_not_hidden_by_return_to_line(self):
        path = StraightnessSummary((0, 0, 0.4), yaw_quaternion(0))
        path.add((1, -0.8, 0.4), yaw_quaternion(-20))
        path.add((2, 0, 0.4), yaw_quaternion(0))
        result = path.result()
        self.assertEqual(result["endpoint_lateral_deviation_m"], 0)
        self.assertEqual(result["maximum_absolute_lateral_deviation_m"], 0.8)
        self.assertGreater(result["horizontal_path_length_m"], result["along_track_progress_m"])
        self.assertAlmostEqual(result["maximum_absolute_heading_deviation_wrapped_degrees"], 20)

    def test_quaternion_sign_does_not_change_heading(self):
        q = yaw_quaternion(73)
        normal = projected_horizontal_forward(q)
        flipped = projected_horizontal_forward(tuple(-value for value in q))
        self.assertEqual(normal, flipped)

    def test_wrap_boundary_and_full_turn_are_distinct(self):
        boundary = StraightnessSummary((0, 0, 0.4), yaw_quaternion(179))
        point = boundary.add((0, 0, 0.4), yaw_quaternion(-179))
        self.assertAlmostEqual(math.degrees(point["heading_drift_wrapped_rad"]), 2)
        self.assertAlmostEqual(math.degrees(point["heading_drift_unwrapped_rad"]), 2)
        turn = StraightnessSummary((0, 0, 0.4), yaw_quaternion(0))
        for yaw in (90, 179, -90, 0):
            turn.add((0, 0, 0.4), yaw_quaternion(yaw))
        result = turn.result()
        self.assertEqual(result["along_track_progress_m"], 0)
        self.assertAlmostEqual(result["endpoint_heading_drift_wrapped_degrees"], 0)
        self.assertAlmostEqual(result["endpoint_heading_drift_unwrapped_degrees"], 360)
        self.assertAlmostEqual(result["maximum_absolute_heading_deviation_unwrapped_degrees"], 360)

    def test_vertical_forward_is_invalid_and_breaks_unwrapping(self):
        vertical = (0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4))
        invalid_anchor = StraightnessSummary((0, 0, 0.4), vertical)
        point = invalid_anchor.add((1, 1, 0.4), yaw_quaternion(0))
        self.assertIsNone(point["along_track_progress_m"])
        self.assertIsNone(invalid_anchor.result()["endpoint_heading_drift_wrapped_rad"])
        path = StraightnessSummary((0, 0, 0.4), yaw_quaternion(0))
        self.assertFalse(path.add((0, 0, 0.4), vertical)["heading_projection_valid"])
        point = path.add((1, 0, 0.4), yaw_quaternion(10))
        self.assertAlmostEqual(math.degrees(point["heading_drift_wrapped_rad"]), 10)
        self.assertIsNone(point["heading_drift_unwrapped_rad"])
        self.assertIsNone(path.result()["maximum_absolute_heading_deviation_unwrapped_rad"])
        self.assertEqual(path.result()["invalid_heading_physics_samples"], 1)

    def test_reset_segments_use_new_real_origin_and_forward(self):
        first = StraightnessSummary((0, 0, 0.4), yaw_quaternion(0))
        first.add((2, 0, 0.4), yaw_quaternion(0))  # Actual terminal sample.
        second = StraightnessSummary((1000, -500, 0.4), yaw_quaternion(90))
        self.assertEqual(second.origin_point["horizontal_path_length_m"], 0)
        second.add((1000, -497, 0.4), yaw_quaternion(90))
        summary = summarize_straightness([{"straightness": first.result()}, {"straightness": second.result()}])
        self.assertAlmostEqual(summary["sum_episode_along_track_progress_m"], 5)
        self.assertAlmostEqual(first.result()["horizontal_path_length_m"] + second.result()["horizontal_path_length_m"], 5)
        self.assertAlmostEqual(summary["maximum_absolute_lateral_deviation_m"], 0)
        self.assertEqual(first.result()["endpoint_world_position_m"], [2, 0, 0.4])

    def test_bad_pose_is_rejected_instead_of_becoming_zero_error(self):
        with self.assertRaises(ValueError):
            projected_horizontal_forward((0, 0, 0, 0))
        with self.assertRaises(ValueError):
            StraightnessSummary((math.nan, 0, 0), yaw_quaternion(0))


if __name__ == "__main__":
    unittest.main()
