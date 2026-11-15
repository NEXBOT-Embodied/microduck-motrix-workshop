"""Protect meaningful classroom contracts without running training or faking weights."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from microduck_course.cli import main
from microduck_course.config import CourseError, evaluation_settings, load_settings, resolved_algorithm, resolve_path, validate_student
from microduck_course.preflight import inspect_mjcf_assets
from microduck_course.runtime import add_upstream_sources, build_train_config, require_learning_progress
from microduck_course.storage import new_run, open_course_run, sha256, validate_export_selection, validate_policy_files, write_json

ROOT = Path(__file__).resolve().parents[1]


class ClassroomContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="microduck-contract-")
        self.root = Path(self.temporary.name)
        shutil.copytree(ROOT / "configs", self.root / "configs")
        teacher = yaml.safe_load((self.root / "configs/classroom.yaml").read_text(encoding="utf-8"))
        teacher["course"]["upstream_path"] = str(ROOT / "upstream/MotrixLab")
        (self.root / "configs/classroom.yaml").write_text(yaml.safe_dump(teacher, allow_unicode=True), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def settings(self):
        return load_settings(self.root, "configs/classroom.yaml", "configs/experiment.yaml")

    def call(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([*args, "--root", str(self.root)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_student_path_isolation_and_unique_runs(self):
        settings = self.settings()
        first, one = new_run(self.root, settings, "s01")
        second, two = new_run(self.root, settings, "s01")
        third, _ = new_run(self.root, settings, "s02")
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent.name, "s01")
        self.assertEqual(third.parent.name, "s02")
        self.assertNotEqual(one["run_id"], two["run_id"])
        for value in ("../s02", "s01/x", "..", "a b", "s01\\.."): 
            with self.assertRaises(CourseError):
                validate_student(value)

    def test_only_one_experiment_factor_and_finite_values(self):
        target = self.root / "configs/experiment.yaml"
        for data in ({"action_rate_multiplier": 1.0, "num_envs": 4096}, {"action_rate_multiplier": float("nan")}, {"action_rate_multiplier": True}, {"action_rate_multiplier": -1}):
            target.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaises(CourseError):
                self.settings()

    def test_dry_run_never_creates_training_output(self):
        code, output, _ = self.call("train", "--student", "s01", "--dry-run")
        data = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(data["status"], "dry_run")
        self.assertFalse(data["training_executed"])
        self.assertFalse(data["checkpoint_created"])
        self.assertFalse((self.root / "outputs").exists())

    def test_missing_starter_blocks_before_run_creation(self):
        code, _, error = self.call("train", "--student", "s01")
        self.assertEqual(code, 2)
        self.assertIn("starter", error)
        self.assertFalse((self.root / "outputs").exists())

    def test_training_failure_is_recorded_and_cannot_be_played(self):
        with patch("microduck_course.runtime.train_real", side_effect=CourseError("expected build-instance failure")):
            code, _, _ = self.call("train", "--student", "s01", "--from-scratch")
        self.assertEqual(code, 2)
        student_outputs = resolve_path(self.root, self.settings()["paths"]["outputs"]) / "s01"
        manifests = list(student_outputs.glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(manifest["walking_verified"])
        with self.assertRaises(CourseError):
            open_course_run(self.root, self.settings(), "s01", manifest["run_id"])

    def test_empty_checkpoint_is_rejected(self):
        path = self.root / "empty-checkpoint.pt"
        path.touch()
        with self.assertRaises(CourseError):
            validate_policy_files(self.root, path, "0" * 64)

    def test_starter_identity_is_written_before_training_and_survives_failure(self):
        source = self.root / "fixture-source"
        source.mkdir()
        checkpoint = source / "fixture-bytes.bin"
        checkpoint.write_bytes(b"unit-test-only provenance fixture, never loaded as policy")
        metadata = {"env_name": "microduck-walk-flat", "source_annotation": "preserve original metadata"}
        write_json(source / "metadata.json", metadata)
        (source / "task_config.yaml").write_text("fixture: not-a-training-run\n", encoding="utf-8")

        def fail_after_reading_manifest(root, settings, path, *_):
            recorded = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["status"], "running")
            self.assertEqual(recorded["start"]["checkpoint_sha256"], sha256(checkpoint))
            self.assertEqual(recorded["start"]["original_metadata"], metadata)
            raise CourseError("intentional failure before any training")

        with patch("microduck_course.cli.resolve_registered_policy", return_value=(source, checkpoint, metadata)), patch("microduck_course.runtime.train_real", side_effect=fail_after_reading_manifest):
            code, _, _ = self.call("train", "--student", "s01")
        self.assertEqual(code, 2)
        student_outputs = resolve_path(self.root, self.settings()["paths"]["outputs"]) / "s01"
        record = json.loads(next(student_outputs.glob("*/manifest.json")).read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["start"]["role"], "starter")
        self.assertEqual(record["start"]["metadata_sha256"], sha256(source / "metadata.json"))
        self.assertEqual(record["start"]["task_config_sha256"], sha256(source / "task_config.yaml"))

    def test_export_selection_rejects_pointer_or_content_drift(self):
        first, other = self.root / "first.bin", self.root / "other.bin"
        first.write_bytes(b"unit-test fixture")
        other.write_bytes(first.read_bytes())
        digest = sha256(first)
        validate_export_selection(first, first, digest)
        with self.assertRaises(CourseError):
            validate_export_selection(other, first, digest)
        first.write_bytes(b"changed fixture")
        with self.assertRaises(CourseError):
            validate_export_selection(first, first, digest)

    def test_sampling_or_critic_only_progress_cannot_count_as_policy_training(self):
        with self.assertRaises(CourseError):
            require_learning_progress(100, 100, 4)
        with self.assertRaises(CourseError):
            require_learning_progress(101, 104, 4)
        evidence = require_learning_progress(103, 105, 4)
        self.assertEqual(evidence["new_critic_updates"], 2)
        self.assertEqual(evidence["new_actor_update_gates"], 1)

    def test_official_task_recipe_is_merged_before_teacher_overrides(self):
        settings = self.settings()
        upstream = Path(settings["course"]["upstream_path"])
        recipe = yaml.safe_load((upstream / "configs/task/microduck-walk-flat/motrix.fastsac.yaml").read_text(encoding="utf-8"))
        algorithm = resolved_algorithm(self.root, settings)
        # Inherit actual task values rather than copying expected numeric literals.
        self.assertEqual(algorithm["agent"]["alpha_init"], recipe["algo"]["agent"]["alpha_init"])
        self.assertEqual(algorithm["agent"]["target_entropy_ratio"], recipe["algo"]["agent"]["target_entropy_ratio"])
        self.assertEqual(algorithm["agent"]["batch_size"], settings["fastsac"]["agent"]["batch_size"])

    def test_config_exposes_expectations_without_claiming_runtime(self):
        code, output, _ = self.call("config")
        report = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(report["derived"]["status"], "static_expectation_not_runtime_measurement")
        contract = report["policy_contract"]
        self.assertEqual(sum(term["size"] for term in contract["policy_observation"]["ordered_terms"]), report["derived"]["expected_static_spaces"]["policy"])
        self.assertEqual(report["derived"]["expected_action_rate_weight"], contract["reward_experiment"]["statically_derived_base_weight"] * self.settings()["experiment"]["action_rate_multiplier"])

    def test_asset_check_detects_lfs_placeholder(self):
        (self.root / "model.xml").write_text('<mujoco><asset><mesh name="duck" file="duck.stl"/></asset></mujoco>', encoding="utf-8")
        (self.root / "duck.stl").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 4\n", encoding="utf-8")
        report = inspect_mjcf_assets(self.root / "model.xml")
        self.assertEqual(report["status"], "assets_blocked")
        self.assertEqual(len(report["lfs_pointers"]), 1)


class OfficialTaskIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings(ROOT, "configs/classroom.yaml", "configs/experiment.yaml")
        add_upstream_sources(ROOT, cls.settings)
        try:
            from motrix_envs.locomotion.humanoid.microduck import make_microduck_walk_flat_cfg
        except ImportError as exc:
            raise unittest.SkipTest(f"Official config dependencies unavailable; runtime not tested: {exc}")
        cls.make_base = staticmethod(make_microduck_walk_flat_cfg)

    def test_multiplier_changes_real_dataclass_without_mutating_base(self):
        from microduck_course.task import apply_experiment
        base = self.make_base()
        baseline_weight = base.rewards.penalty_action_rate.weight
        tracking_weight = base.rewards.tracking_lin_vel.weight
        actual = apply_experiment(base, 2.0)
        self.assertEqual(actual.rewards.penalty_action_rate.weight, baseline_weight * 2.0)
        self.assertEqual(base.rewards.penalty_action_rate.weight, baseline_weight)
        self.assertEqual(actual.rewards.tracking_lin_vel.weight, tracking_weight)
        self.assertEqual(self.make_base().rewards.penalty_action_rate.weight, baseline_weight)

    def test_real_registry_factory_reads_saved_experiment(self):
        from microduck_course.task import register
        from motrix_env_core import registry
        register()
        with tempfile.TemporaryDirectory(prefix="microduck-real-config-") as temporary:
            target = Path(temporary) / "snapshot.json"
            data = copy.deepcopy(self.settings)
            data["experiment"]["action_rate_multiplier"] = 2.0
            write_json(target, data)
            with patch.dict(os.environ, {"MICRODUCK_COURSE_ACTIVE_CONFIG": str(target)}):
                actual = registry.make_env_config("microduck-course-flat")
        self.assertEqual(actual.rewards.penalty_action_rate.weight, self.make_base().rewards.penalty_action_rate.weight * 2)

    def test_evaluation_pins_real_command_without_changing_training_distribution(self):
        from microduck_course.task import register
        from motrix_env_core import registry
        register()
        base = self.make_base()
        with tempfile.TemporaryDirectory(prefix="microduck-evaluation-config-") as temporary:
            snapshot = Path(temporary) / "snapshot.json"
            write_json(snapshot, self.settings)
            with patch.dict(os.environ, {"MICRODUCK_COURSE_ACTIVE_CONFIG": str(snapshot)}):
                training = registry.make_env_config("microduck-course-flat")
                write_json(snapshot, evaluation_settings(self.settings))
                evaluation = registry.make_env_config("microduck-course-flat")
        expected = tuple(float(v) for v in self.settings["evaluation"]["command"])
        self.assertEqual(training.commands.walk.vel_limit, base.commands.walk.vel_limit)
        self.assertEqual(training.commands.walk.stand_prob, base.commands.walk.stand_prob)
        self.assertEqual(evaluation.commands.walk.vel_limit, (expected, expected))
        self.assertEqual(evaluation.commands.walk.stand_prob, 0.0)
        self.assertNotEqual(base.commands.walk.vel_limit, evaluation.commands.walk.vel_limit)

    def test_reference_playback_uses_course_evaluation_context_without_editing_source(self):
        from types import SimpleNamespace
        from microduck_course.runtime import play_real
        from microduck_course.task import register
        from motrix_rl import runs
        from motrix_rl_builtin.fastsac.config import FastSacCfg
        register()
        with tempfile.TemporaryDirectory(prefix="microduck-playback-context-") as temporary:
            source = Path(temporary)
            run = runs.create_run_context("microduck-walk-flat", "motrix", "torch", "fastsac", 42, "pt", runs_root=source, sim="motrixsim")
            config, _ = build_train_config(ROOT, self.settings, None, None)
            config.task.env = "microduck-walk-flat"
            runs.write_task_config(run.run_dir, config)
            original_metadata = (run.run_dir / "metadata.json").read_bytes()
            original_task = (run.run_dir / "task_config.yaml").read_bytes()
            snapshot = source / "evaluation.json"
            playback = evaluation_settings(self.settings)
            write_json(snapshot, playback)
            fake_handle = SimpleNamespace(play=lambda policy: None)
            with patch.dict(os.environ, {"MICRODUCK_COURSE_ACTIVE_CONFIG": str(snapshot)}), patch("microduck_course.runtime.setup_runtime"), patch("motrix_rl.frameworks.get_agent_provider", return_value=SimpleNamespace(config_type=FastSacCfg)), patch("motrix_rl.frameworks.create_trainer", return_value=fake_handle) as create:
                report = play_real(ROOT, playback, snapshot, run.run_dir, source / "not-loaded-as-policy.pt")
            context = create.call_args.args[0]
            self.assertEqual(context.env_name, "microduck-course-flat")
            self.assertEqual(context.seed, self.settings["evaluation"]["seed"])
            self.assertEqual(report["source_env"], "microduck-walk-flat")
            self.assertEqual(report["actual_command_bounds"][0], self.settings["evaluation"]["command"])
            self.assertFalse(report["velocity_metrics_computed"])
            self.assertEqual((run.run_dir / "metadata.json").read_bytes(), original_metadata)
            self.assertEqual((run.run_dir / "task_config.yaml").read_bytes(), original_task)

    def test_real_typed_fastsac_config_and_non_noop_resume_target(self):
        from motrix_rl_builtin.fastsac.config import FastSacCfg
        initial_config, initial_step = build_train_config(ROOT, self.settings, None, None)
        self.assertIsInstance(initial_config.algo, FastSacCfg)
        self.assertEqual(initial_step, 0)
        with tempfile.TemporaryDirectory(prefix="microduck-starter-config-") as temporary:
            starter = Path(temporary)
            actual_algo = resolved_algorithm(ROOT, self.settings)
            actual_algo["agent"]["actor_hidden_dim"] += 16
            (starter / "task_config.yaml").write_text(yaml.safe_dump({"algo": actual_algo}), encoding="utf-8")
            # Only checkpoint deserialization is mocked; typed config composition is the real API.
            with patch("microduck_course.runtime.checkpoint_state", return_value={"global_step": 900}):
                config, step = build_train_config(ROOT, self.settings, starter, starter / "not-loaded.pt")
        self.assertEqual(step, 900)
        self.assertEqual(config.algo.trainer.num_learning_iterations, step + self.settings["runtime"]["additional_iterations"])
        self.assertEqual(config.algo.agent.actor_hidden_dim, actual_algo["agent"]["actor_hidden_dim"])
        self.assertEqual(config.algo.agent.alpha_init, initial_config.algo.agent.alpha_init)


if __name__ == "__main__":
    unittest.main()
