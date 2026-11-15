"""Protect the editable-YAML → real task and immutable replay-snapshot contracts."""

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microduck_course.config import CourseError, evaluation_settings, load_settings
from microduck_course.constraints import apply_constraints, attach_constraints
from microduck_course.runtime import add_upstream_sources
from microduck_course.storage import write_json

ROOT = Path(__file__).resolve().parents[1]


class WalkingConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings(ROOT, "configs/classroom.yaml", "configs/experiment.yaml")
        add_upstream_sources(ROOT, cls.settings)
        try:
            from motrix_envs.locomotion.humanoid.microduck import make_microduck_walk_flat_cfg
        except ImportError as exc:
            raise unittest.SkipTest(f"需要官方任务配置的轻依赖：{exc}")
        cls.make_base = staticmethod(make_microduck_walk_flat_cfg)

    def setUp(self):
        from microduck_course.task import apply_teacher_task_profile
        self.settings = attach_constraints(ROOT, self.settings, "configs/walking_constraints.yaml")
        self.data = copy.deepcopy(self.settings["walking_constraints"])
        self.base = apply_teacher_task_profile(self.make_base(), self.settings["teacher_task_profile"])

    def test_edit_changes_real_reward_and_command_without_mutating_source(self):
        reward = self.data["task"]["rewards"]["penalty_heading_rate"]
        reward["weight"] *= 1.1
        self.data["task"]["commands"]["walk"]["vel_limit"][0][0] *= 0.9
        actual = apply_constraints(self.base, self.data)
        self.assertEqual(actual.rewards.penalty_heading_rate.weight, reward["weight"])
        self.assertNotEqual(actual.rewards.penalty_heading_rate.weight, self.base.rewards.penalty_heading_rate.weight)
        self.assertNotEqual(actual.commands.walk.vel_limit, self.base.commands.walk.vel_limit)

    def test_unknown_key_fails_instead_of_being_ignored(self):
        self.data["task"]["rewards"]["penalty_heading_rate"]["wieght"] = -1
        with self.assertRaises(CourseError):
            apply_constraints(self.base, self.data)

    def test_nonfinite_and_wrong_sign_rewards_are_rejected(self):
        for bad in (float("nan"), float("inf"), True, abs(self.base.rewards.penalty_heading_rate.weight)):
            with self.subTest(value=bad):
                data = copy.deepcopy(self.data)
                data["task"]["rewards"]["penalty_heading_rate"]["weight"] = bad
                with self.assertRaises(CourseError):
                    apply_constraints(self.base, data)

    def test_control_period_and_observation_layout_cannot_silently_disagree(self):
        data = copy.deepcopy(self.data)
        data["task"]["ctrl_dt"] *= 2
        with self.assertRaises(CourseError):
            apply_constraints(self.base, data)
        self.data["task"]["observations"]["policy"]["sin_phase"]["size"] += 1
        with self.assertRaises(CourseError):
            apply_constraints(self.base, self.data)

    def test_saved_snapshot_drives_training_and_replay_without_multiplier_stacking(self):
        from microduck_course.task import make_course_cfg
        from motrix_envs.locomotion.humanoid.microduck import make_microduck_walk_flat_cfg
        weight = self.data["task"]["rewards"]["penalty_action_rate"]["weight"] * 1.3
        self.settings["walking_constraints"]["task"]["rewards"]["penalty_action_rate"]["weight"] = weight
        self.settings["experiment"]["action_rate_multiplier"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "course_config.json"
            write_json(snapshot, self.settings)
            # Editing the live in-memory values later cannot rewrite the saved snapshot.
            self.settings["walking_constraints"]["task"]["rewards"]["penalty_action_rate"]["weight"] *= 2
            with patch.dict(os.environ, {"MICRODUCK_COURSE_ACTIVE_CONFIG": str(snapshot)}):
                actual = make_course_cfg()
                self.assertEqual(actual.rewards.penalty_action_rate.weight, weight)
                replay = evaluation_settings(self.settings)
                write_json(snapshot, replay)
                replayed = make_course_cfg()
                command = tuple(replay["evaluation"]["command"])
                self.assertEqual(replayed.commands.walk.vel_limit, (command, command))
        self.assertNotEqual(actual.commands.walk.vel_limit, make_microduck_walk_flat_cfg().commands.walk.vel_limit)


if __name__ == "__main__":
    unittest.main()
