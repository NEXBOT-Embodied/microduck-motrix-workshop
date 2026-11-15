"""Load the teacher's annotated task settings and apply them to real MotrixLab cfgs."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path

from microduck_course.config import CourseError, read_yaml, resolve_path

TASK_FIELDS = ("max_episode_seconds", "ctrl_dt", "sim", "commands", "rewards",
               "actions", "sim_reset", "observations", "terminations")


def load_constraints(root: Path, path: str) -> dict:
    data = read_yaml(resolve_path(root, path))
    if set(data) != {"schema_version", "task", "evaluation"} or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise CourseError("约束 YAML 需要 schema_version: 1、task 和 evaluation 三部分。")
    if not isinstance(data["task"], dict) or set(data["task"]) != set(TASK_FIELDS):
        raise CourseError("task 必须完整包含：" + "、".join(TASK_FIELDS))
    evaluation = data["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {"command"}:
        raise CourseError("evaluation 仅支持 command: [前进速度, 横移速度, 转向速度]。")
    command = evaluation["command"]
    if not isinstance(command, list) or len(command) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in command
    ):
        raise CourseError("evaluation.command 必须包含三个有限数字。")
    return data


def attach_constraints(root: Path, settings: dict, path: str) -> dict:
    """Embed values, rather than a live file reference, so playback stays reproducible."""
    result = copy.deepcopy(settings)
    # The complete YAML includes the added heading reward; use its typed task
    # structure, then let YAML replace every numerical task setting.
    result["teacher_task_profile"] = "forward_heading_rate_mid_v1"
    result["walking_constraints"] = load_constraints(root, path)
    result["evaluation"]["command"] = list(result["walking_constraints"]["evaluation"]["command"])
    return result


def _patch(old, value, path: str):
    if path.startswith("task.observations.") and path.rsplit(".", 1)[-1] in ("offset", "size") and value != old:
        raise CourseError(f"{path} 决定观测维度，续训旧模型时保持 {old}。")
    if is_dataclass(old) or isinstance(old, dict):
        baseline = {f.name: getattr(old, f.name) for f in fields(old)} if is_dataclass(old) else old
        if not isinstance(value, dict) or set(value) != set(baseline):
            raise CourseError(f"{path} 的字段不完整或拼写有误；应为：{', '.join(baseline)}")
        updated = {key: _patch(item, value[key], f"{path}.{key}") for key, item in baseline.items()}
        return replace(old, **updated) if is_dataclass(old) else updated
    if isinstance(old, tuple):
        if not isinstance(value, list) or len(value) != len(old):
            raise CourseError(f"{path} 必须包含 {len(old)} 项。")
        return tuple(_patch(item, new, f"{path}[{i}]") for i, (item, new) in enumerate(zip(old, value)))
    if isinstance(old, bool):
        if not isinstance(value, bool):
            raise CourseError(f"{path} 只能填 true 或 false。")
    elif isinstance(old, (float, int)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise CourseError(f"{path} 必须是有限数字。")
        if isinstance(old, int) and not isinstance(value, int):
            raise CourseError(f"{path} 必须是整数。")
        return type(old)(value)
    elif old is None and value is not None:
        sizes = {"task.sim.gravity": 3, "task.sim_reset.humanoid_state.randomization.sliding_friction_range": 2}
        size = sizes.get(path)
        if size is None or not isinstance(value, list) or len(value) != size:
            raise CourseError(f"{path} 不是本课程开放修改的字段。")
        return tuple(_patch(0.0, x, f"{path}[{i}]") for i, x in enumerate(value))
    elif value != old:
        raise CourseError(f"{path} 是模型或观测标识，保持 {old!r}，不要重命名。")
    return value


def apply_constraints(cfg, data: dict):
    """Override numerical knobs while preserving observation/action names and dimensions."""
    values = data["task"]
    changes = {key: _patch(getattr(cfg, key), values[key], f"task.{key}") for key in TASK_FIELDS}
    result = replace(cfg, **changes)
    walk = result.commands.walk
    if result.sim.dt <= 0 or result.ctrl_dt <= 0 or result.max_episode_seconds <= 0:
        raise CourseError("仿真步长、控制周期、回合时长必须大于 0。")
    ratio = result.ctrl_dt / result.sim.dt
    if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=1e-8):
        raise CourseError("ctrl_dt 必须是 sim.dt 的正整数倍。")
    if values["commands"]["walk"]["ctrl_dt"] != result.ctrl_dt:
        raise CourseError("commands.walk.ctrl_dt 必须与 task.ctrl_dt 相同。")
    if result.sim.solver_iterations < 1 or result.sim.solver_tolerance <= 0:
        raise CourseError("求解次数必须是正整数，求解容差必须大于 0。")
    if walk.resampling_time < result.ctrl_dt or walk.gait_period <= 0 or not 0 <= walk.stand_prob <= 1:
        raise CourseError("检查指令刷新时间、步态周期和站立概率（0–1）。")
    if not 0 <= walk.gait_period_randomization_width < walk.gait_period:
        raise CourseError("步态周期的随机变化幅度必须非负且小于步态周期。")
    if any(lo > hi for lo, hi in zip(*walk.vel_limit)):
        raise CourseError("vel_limit 的下限不能高于上限。")
    if not 0 <= walk.min_scale <= walk.initial_scale <= walk.max_scale <= 1 or not 0 <= walk.degree < 1:
        raise CourseError("惩罚课程的倍率需满足 0 ≤ min ≤ initial ≤ max ≤ 1；degree 在 [0, 1)。")
    if not 0 <= walk.level_down_threshold <= walk.level_up_threshold:
        raise CourseError("惩罚课程的升级门槛不能低于降级门槛。")
    action = result.actions.joint_position
    lo, hi = action.action_delay_steps
    if action.action_scale <= 0 or lo < 0 or hi < lo:
        raise CourseError("动作幅度必须大于 0；延迟需满足 0 ≤ 下限 ≤ 上限。")
    if result.sim_reset.humanoid_state.spawn_xy_range < 0:
        raise CourseError("出生位置随机范围不能为负。")
    for term in fields(result.rewards):
        reward = getattr(result.rewards, term.name)
        base = getattr(cfg.rewards, term.name)
        if (base.weight < 0 and reward.weight > 0) or (base.weight > 0 and reward.weight < 0):
            raise CourseError(f"{term.name}.weight 的正负号反了；填 0 可以关闭该项。")
        for key in ("sigma", "feet_phase_sigma", "swing_height", "close_feet_threshold"):
            if hasattr(reward, key) and getattr(reward, key) <= 0:
                raise CourseError(f"{term.name}.{key} 必须大于 0。")
    if any(x < 0 for x in result.rewards.pose.pose_weights.values()):
        raise CourseError("pose_weights 各关节的姿态权重不能为负。")
    for group in (result.observations.policy, result.observations.value):
        for term in fields(group):
            noise = getattr(getattr(group, term.name), "noise", None)
            if noise is not None and noise.amplitude < 0:
                raise CourseError("观测噪声幅度不能为负。")
    return result


def constraint_summary(cfg) -> dict:
    """The exact supported task surface; does not build or step a simulator."""
    return {key: asdict(getattr(cfg, key)) if is_dataclass(getattr(cfg, key)) else getattr(cfg, key)
            for key in TASK_FIELDS}
