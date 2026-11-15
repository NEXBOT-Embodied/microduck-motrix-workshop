"""Register one real task variant, derived from the official MicroDuck task."""

from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingManagerEnvCfg

from microduck_course.config import COURSE_TASK, validate_teacher_task_profile

_registered = False


def apply_teacher_task_profile(base_cfg, profile: str | None):
    """Apply an opt-in teacher trial without changing legacy task defaults."""
    profile = validate_teacher_task_profile(profile)
    if profile is None:
        return base_cfg
    if profile in ("forward_heading_rate_strong_v1", "forward_heading_rate_mid_v1"):
        # Controlled continuation: inherit the existing heading profile and
        # change only this bounded cost's weight, keeping its sigma unchanged.
        heading = apply_teacher_task_profile(base_cfg, "forward_heading_rate_v1")
        return replace(heading, rewards=replace(
            heading.rewards,
            penalty_heading_rate=replace(
                heading.rewards.penalty_heading_rate,
                weight=-25.0 if profile == "forward_heading_rate_strong_v1" else -20.0,
            ),
        ))
    if profile in ("forward_walk_v1", "forward_straight_sigma_v1", "forward_heading_rate_v1"):
        walk = replace(
            base_cfg.commands.walk,
            vel_limit=((0.12, 0.0, 0.0), (0.25, 0.0, 0.0)),
            stand_prob=0.0,
        )
        rewards = replace(
            base_cfg.rewards,
            tracking_lin_vel=replace(base_cfg.rewards.tracking_lin_vel, weight=18.0, sigma=0.02),
            alive=replace(base_cfg.rewards.alive, weight=2.0),
            feet_phase=replace(base_cfg.rewards.feet_phase, weight=6.0),
        )
        if profile == "forward_straight_sigma_v1":
            # Controlled teacher trial: only tighten the existing angular tracking reward.
            # Upstream formula is exp(-squared_error / sigma), not / sigma**2.
            rewards = replace(rewards, tracking_ang_vel=replace(rewards.tracking_ang_vel, sigma=0.015))
        if profile == "forward_heading_rate_v1":
            from microduck_course.heading_rewards import HeadingConstrainedRewardsCfg
            rewards = HeadingConstrainedRewardsCfg(**{item.name: getattr(rewards, item.name) for item in fields(rewards)})
        return replace(base_cfg, commands=replace(base_cfg.commands, walk=walk), rewards=rewards)
    raise AssertionError("validated teacher profile has no implementation")


def apply_experiment(base_cfg, multiplier: float):
    """Replace the actual task reward dataclass, preserving all other terms."""
    term = base_cfg.rewards.penalty_action_rate
    return replace(base_cfg, rewards=replace(base_cfg.rewards, penalty_action_rate=replace(term, weight=term.weight * multiplier)))


def apply_evaluation(base_cfg, evaluation: dict):
    """Only playback pins the actual WalkCommandCfg sampling distribution."""
    command = tuple(float(value) for value in evaluation["command"])
    return replace(base_cfg, commands=replace(base_cfg.commands, walk=replace(base_cfg.commands.walk, vel_limit=(command, command), stand_prob=0.0)))


def make_course_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    from motrix_envs.locomotion.humanoid.microduck import make_microduck_walk_flat_cfg

    snapshot = os.environ.get("MICRODUCK_COURSE_ACTIVE_CONFIG")
    if not snapshot:
        raise RuntimeError("请通过 microduck_course 命令运行，以便保存并加载正确的实验配置。")
    data = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    result = apply_teacher_task_profile(make_microduck_walk_flat_cfg(), data.get("teacher_task_profile"))
    if "walking_constraints" in data:
        from microduck_course.constraints import apply_constraints
        result = apply_constraints(result, data["walking_constraints"])
    else:
        result = apply_experiment(result, data["experiment"]["action_rate_multiplier"])
    if data.get("execution_mode") == "evaluation":
        result = apply_evaluation(result, data["evaluation"])
    return result


def register() -> None:
    global _registered
    if _registered:
        return
    from motrix_env_core import registry
    from motrix_env_core.manager import ManagerEnv
    from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingManagerEnvCfg

    registry.register_env_config(COURSE_TASK, make_course_cfg, cfg_type=HumanoidVelocityTrackingManagerEnvCfg)
    registry.env(COURSE_TASK)(ManagerEnv)
    _registered = True
