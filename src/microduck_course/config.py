"""Small, validated classroom configuration surface."""

from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path

import yaml

UPSTREAM_COMMIT = "1ddb9743c220c3a1e234ea0237d5ce9cc56d0499"
COURSE_TASK = "microduck-course-flat"
BASE_TASK = "microduck-walk-flat"
TEACHER_TASK_PROFILES = ("forward_walk_v1", "forward_straight_sigma_v1", "forward_heading_rate_v1", "forward_heading_rate_strong_v1", "forward_heading_rate_mid_v1")


class CourseError(ValueError):
    pass


def read_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise CourseError(f"不能读取 YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CourseError(f"配置必须是映射: {path}")
    return value


def project_root(explicit: str | None = None) -> Path:
    return Path(explicit or os.environ.get("MICRODUCK_COURSE_ROOT") or Path(__file__).resolve().parents[2]).resolve()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_student(student: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", student):
        raise CourseError("学员代号仅可用 1–40 位字母、数字、下划线、短横线，不能含路径或真实个人信息。")
    return student


def validate_teacher_task_profile(profile: object) -> str | None:
    """Only named, versioned teacher profiles may alter the official task."""
    if profile is None:
        return None
    if not isinstance(profile, str) or profile not in TEACHER_TASK_PROFILES:
        raise CourseError(f"teacher_task_profile 仅支持 {', '.join(TEACHER_TASK_PROFILES)}；省略或 null 保持原任务。")
    return profile


def training_visualization_options(value: object = False) -> dict:
    """Validate the optional, bounded native training viewer/recording surface."""
    defaults = {
        "enabled": False, "record": True, "render_every_steps": 2,
        "record_fps": 15, "width": 960, "height": 540,
        "record_every_iterations": 1000, "clip_iterations": 250,
        "max_recorded_frames": 1500, "realtime": True, "startup_hold_seconds": 0.0,
    }
    if isinstance(value, bool):
        return {**defaults, "enabled": value}
    if not isinstance(value, dict) or set(value) - defaults.keys():
        raise CourseError("training_visualization 必须是布尔值或仅含已支持字段的映射。")
    result = {**defaults, **value}
    for key in ("enabled", "record", "realtime"):
        if not isinstance(result[key], bool):
            raise CourseError(f"training_visualization.{key} 必须是布尔值。")
    limits = {"render_every_steps": (1, 100), "record_fps": (1, 50),
              "width": (320, 1920), "height": (240, 1080),
              "record_every_iterations": (0, 1000000), "clip_iterations": (1, 100000),
              "max_recorded_frames": (1, 10000)}
    for key, (minimum, maximum) in limits.items():
        item = result[key]
        if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
            raise CourseError(f"training_visualization.{key} 必须是 {minimum}–{maximum} 的整数。")
    if result["width"] % 2 or result["height"] % 2:
        raise CourseError("training_visualization 录制宽高必须为偶数。")
    period = result["record_every_iterations"]
    if period and result["clip_iterations"] > period:
        raise CourseError("training_visualization.clip_iterations 不能超过阶段间隔。")
    hold = result["startup_hold_seconds"]
    if isinstance(hold, bool) or not isinstance(hold, (int, float)) or not math.isfinite(hold) or not 0 <= hold <= 30:
        raise CourseError("training_visualization.startup_hold_seconds 必须是 0–30 秒有限数字。")
    return result


def load_settings(root: Path, classroom: str, experiment: str) -> dict:
    teacher = read_yaml(resolve_path(root, classroom))
    validate_teacher_task_profile(teacher.get("teacher_task_profile"))
    training_visualization_options(teacher.get("training_visualization", False))
    experiment_data = read_yaml(resolve_path(root, experiment))
    if set(experiment_data) != {"action_rate_multiplier"}:
        raise CourseError("experiment.yaml 只允许 action_rate_multiplier；教师资源参数位于 classroom.yaml。")
    multiplier = experiment_data["action_rate_multiplier"]
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier):
        raise CourseError("action_rate_multiplier 必须是有限数字。")
    if not 0.25 <= multiplier <= 4.0:
        raise CourseError("课堂 action_rate_multiplier 范围为 0.25–4.0。")
    course, runtime = teacher["course"], teacher["runtime"]
    if (course["task"], course["base_task"], course["simulator"], course["algorithm"], course["upstream_commit"]) != (
        COURSE_TASK, BASE_TASK, "motrixsim", "motrix.fastsac", UPSTREAM_COMMIT
    ):
        raise CourseError("课程任务/引擎/算法/上游固定版本不匹配。")
    for key in ("num_envs", "play_num_envs", "additional_iterations", "cpu_threads", "logging_interval"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CourseError(f"runtime.{key} 必须是正整数。")
    if isinstance(runtime["checkpoint_interval"], bool) or not isinstance(runtime["checkpoint_interval"], int) or runtime["checkpoint_interval"] < 0:
        raise CourseError("checkpoint_interval 必须是非负整数。")
    seed = runtime["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CourseError("runtime.seed 必须是非负整数。")
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", runtime["device"]):
        raise CourseError("runtime.device 仅支持 auto、cpu、cuda 或 cuda:N。")
    if teacher["fastsac"]["asynchronous"] is not False:
        raise CourseError("课程第一版固定同步 FastSAC，尚未验收异步课堂路径。")
    evaluation = teacher["evaluation"]
    command = evaluation["command"]
    if not isinstance(command, list) or len(command) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in command):
        raise CourseError("evaluation.command 必须是 3 个有限数值 [vx, vy, wz]。")
    if isinstance(evaluation["seed"], bool) or not isinstance(evaluation["seed"], int) or evaluation["seed"] < 0:
        raise CourseError("evaluation.seed 必须是非负整数。")
    settings = copy.deepcopy(teacher)
    settings["experiment"] = {"action_rate_multiplier": float(multiplier)}
    return settings


def evaluation_settings(settings: dict) -> dict:
    result = copy.deepcopy(settings)
    result["execution_mode"] = "evaluation"
    return result


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolved_algorithm(root: Path, settings: dict, starter_run: Path | None = None) -> dict:
    """Follow the official recipe order before applying classroom resource knobs."""
    upstream = resolve_path(root, settings["course"]["upstream_path"])
    base = read_yaml(upstream / "motrix_rl/src/motrix_rl/configs/algo_base/motrix.fastsac.yaml")
    base.pop("defaults", None)
    recipe = read_yaml(upstream / "configs/task/microduck-walk-flat/motrix.fastsac.yaml")
    result = deep_merge(base, recipe.get("algo", {}))
    if starter_run is not None:
        result = deep_merge(result, read_yaml(starter_run / "task_config.yaml")["algo"])
    result = deep_merge(result, settings["fastsac"])
    result["device"] = None if settings["runtime"]["device"] == "auto" else settings["runtime"]["device"]
    result["trainer"]["num_learning_iterations"] = settings["runtime"]["additional_iterations"]
    return result
