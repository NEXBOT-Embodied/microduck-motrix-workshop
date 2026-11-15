"""Save truthful offline evidence; does not run training or simulation."""

from __future__ import annotations

import io
import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microduck_course.config import load_settings, resolved_algorithm
from microduck_course.preflight import inspect_mjcf_assets
from microduck_course.runtime import add_upstream_sources
from microduck_course.storage import write_json


def main() -> int:
    evidence = ROOT / "checks/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "checks"), pattern="test_*.py")
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (evidence / "offline-tests.txt").write_text(log.getvalue(), encoding="utf-8")
    settings = load_settings(ROOT, "configs/classroom.yaml", "configs/experiment.yaml")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "status": "passed" if result.wasSuccessful() else "failed",
        "training_executed": False, "sdk_simulation_executed": False,
        "render_verified": False, "policy_weights_created": False, "capacity_50_users_verified": False,
        "algorithm": resolved_algorithm(ROOT, settings),
        "assets": inspect_mjcf_assets(ROOT / "upstream/MotrixLab/motrix_robots/src/motrix_robots/assets/microduck/microduck.xml"),
    }
    try:
        add_upstream_sources(ROOT, settings)
        from motrix_envs.locomotion.humanoid.microduck import make_microduck_walk_flat_cfg
        from microduck_course.task import apply_evaluation, apply_experiment
        base = make_microduck_walk_flat_cfg()
        actual = apply_experiment(base, 2.0)
        report["actual_task_dataclass"] = {
            "status": "instantiated_without_simulator", "base_action_rate_weight": base.rewards.penalty_action_rate.weight,
            "multiplier": 2.0, "actual_action_rate_weight": actual.rewards.penalty_action_rate.weight,
            "unmodified_tracking_weight": actual.rewards.tracking_lin_vel.weight,
        }
        evaluation = apply_evaluation(base, settings["evaluation"])
        report["actual_evaluation_dataclass"] = {
            "status": "factory_configured_without_simulator",
            "preset": settings["evaluation"],
            "actual_command_bounds": [list(row) for row in evaluation.commands.walk.vel_limit],
            "actual_stand_probability": evaluation.commands.walk.stand_prob,
            "training_command_bounds_preserved": [list(row) for row in base.commands.walk.vel_limit],
            "velocity_metrics_computed": False,
        }
    except ImportError as exc:
        report["actual_task_dataclass"] = {"status": "not_instantiated", "reason": str(exc)}
    write_json(evidence / "offline-acceptance.json", report)
    print(log.getvalue())
    print(json.dumps({key: value for key, value in report.items() if key not in ("algorithm", "assets")}, ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
