"""Measure a real saved FastSAC policy in MotrixSim, without training/rendering.

Usage (from the course root):
  python tools/evaluate_walking.py --run-dir ACTUAL_RUN_DIR \
    --checkpoint checkpoints/model_0001000.pt --settings SAVED_COURSE_CONFIG \
    --seconds 20 --seed 42 --output NEW_OUTPUT_DIRECTORY

The evaluation uses CPU policy inference and the unchanged official task. A local
read-only callback samples physical poses/velocities before automatic resets.
No success threshold or automatic "walking verified" decision is invented here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import traceback
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_TERMS = [
    ("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3),
    ("command", 3), ("joint_pos", 14), ("joint_vel", 14),
    ("actions", 14), ("sin_phase", 2), ("cos_phase", 2),
]
CSV_FIELDS = [
    "control_step", "simulated_seconds", "episode_index", "episode_elapsed_seconds",
    "terminated_collision", "truncated_timeout", "automatic_reset_occurred",
    "post_reset_returned_observation_excluded", "raw_critic_crosscheck_used",
    "body_vx_mps", "body_vy_mps", "body_vz_mps", "body_wx_radps", "body_wy_radps", "body_wz_radps",
    "target_vx_mps", "target_vy_mps", "target_body_wz_radps",
    "vx_error_mps", "vy_error_mps", "body_wz_error_radps", "planar_velocity_error_mps",
    "world_vx_mps", "world_vy_mps", "world_vz_mps",
    "world_wx_radps", "world_wy_radps", "world_wz_radps",
    "base_qx", "base_qy", "base_qz", "base_qw", "transition_reward",
]
TRAJECTORY_FIELDS = [
    "sample_kind", "control_step", "simulated_seconds", "episode_index", "episode_elapsed_seconds",
    "terminated_collision", "truncated_timeout", "automatic_reset_occurred",
    "world_x_m", "world_y_m", "world_z_m", "base_qx", "base_qy", "base_qz", "base_qw",
    "episode_origin_x_m", "episode_origin_y_m", "episode_origin_z_m",
    "initial_forward_x", "initial_forward_y", "initial_left_x", "initial_left_y",
    "initial_heading_rad", "horizontal_forward_projection_norm", "heading_projection_valid",
    "along_track_progress_m", "lateral_deviation_m", "horizontal_path_length_m",
    "heading_drift_wrapped_rad", "heading_drift_unwrapped_rad",
]
HEADING_PROJECTION_EPSILON = 1e-6


def projected_horizontal_forward(quaternion) -> tuple[tuple[float, float] | None, float]:
    """Project rotated local +X onto world XY, using the actual xyzw pose.

    The projection tolerance only detects an undefined horizontal heading; it
    is not a walking/pass criterion. Normalizing the quaternion removes small
    simulator roundoff, after verifying the same unit-quaternion contract as
    the velocity measurement.
    """
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("基座四元数必须为4个有限xyzw值")
    norm_squared = sum(value * value for value in values)
    if not math.isclose(norm_squared, 1.0, abs_tol=1e-3):
        raise ValueError("基座四元数未单位化，不能计算水平朝向")
    x, y, z, w = (value / math.sqrt(norm_squared) for value in values)
    forward_x, forward_y = 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w)
    norm = math.hypot(forward_x, forward_y)
    if norm < HEADING_PROJECTION_EPSILON:
        return None, norm
    return (forward_x / norm, forward_y / norm), norm


def signed_heading_difference(initial, current) -> float:
    return math.atan2(initial[0] * current[1] - initial[1] * current[0],
                      initial[0] * current[0] + initial[1] * current[1])


class StraightnessSummary:
    """One episode's actual path in its fixed initial horizontal body axes.

    Construct a new instance from a read-only reset-pose sample for every
    episode. Only pre-reset physical samples may be passed to add(), so a reset
    teleport can neither count as forward progress nor inflate path length.
    """

    def __init__(self, position, quaternion) -> None:
        self.origin = self._position(position)
        self.initial_quaternion = tuple(float(value) for value in quaternion)
        self.forward, initial_projection_norm = projected_horizontal_forward(quaternion)
        self.left = (-self.forward[1], self.forward[0]) if self.forward else None
        self.initial_heading = math.atan2(self.forward[1], self.forward[0]) if self.forward else None
        self.samples = 0
        self.previous_position = self.origin
        self.path_length = 0.0
        self.maximum_abs_lateral = 0.0
        self.minimum_along = self.maximum_along = 0.0
        self.maximum_abs_heading_wrapped = self.maximum_abs_heading_unwrapped = 0.0
        self.valid_heading_samples = self.invalid_heading_samples = 0
        self.unwrapped_heading = 0.0 if self.forward else None
        self.previous_heading = self.forward
        self.unwrap_contiguous = self.forward is not None
        self.last = self._point(self.origin, self.forward, initial_projection_norm)
        self.origin_point = dict(self.last)

    @staticmethod
    def _position(position) -> tuple[float, float, float]:
        values = tuple(float(value) for value in position)
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("世界位置必须为3个有限XYZ值")
        return values

    def _point(self, position, forward, projection_norm) -> dict:
        displacement = (position[0] - self.origin[0], position[1] - self.origin[1])
        along = sum(a * b for a, b in zip(displacement, self.forward)) if self.forward else None
        lateral = sum(a * b for a, b in zip(displacement, self.left)) if self.left else None
        heading = signed_heading_difference(self.forward, forward) if self.forward and forward else None
        return {
            "episode_origin_x_m": self.origin[0], "episode_origin_y_m": self.origin[1],
            "episode_origin_z_m": self.origin[2],
            "initial_forward_x": self.forward[0] if self.forward else None,
            "initial_forward_y": self.forward[1] if self.forward else None,
            "initial_left_x": self.left[0] if self.left else None,
            "initial_left_y": self.left[1] if self.left else None,
            "initial_heading_rad": self.initial_heading,
            "horizontal_forward_projection_norm": projection_norm,
            "heading_projection_valid": forward is not None,
            "along_track_progress_m": along, "lateral_deviation_m": lateral,
            "horizontal_path_length_m": self.path_length,
            "heading_drift_wrapped_rad": heading,
            "heading_drift_unwrapped_rad": self.unwrapped_heading if self.unwrap_contiguous else None,
        }

    def add(self, position, quaternion) -> dict:
        position = self._position(position)
        forward, projection_norm = projected_horizontal_forward(quaternion)
        self.path_length += math.hypot(position[0] - self.previous_position[0],
                                      position[1] - self.previous_position[1])
        self.previous_position = position
        if self.forward and forward:
            self.valid_heading_samples += 1
            if self.unwrap_contiguous:
                self.unwrapped_heading += signed_heading_difference(self.previous_heading, forward)
                self.maximum_abs_heading_unwrapped = max(self.maximum_abs_heading_unwrapped, abs(self.unwrapped_heading))
            self.previous_heading = forward
        else:
            self.invalid_heading_samples += 1
            # An invalid interval cannot be silently bridged by angle unwrapping.
            self.unwrap_contiguous = False
        self.last = self._point(position, forward, projection_norm)
        self.samples += 1
        if self.forward:
            self.maximum_abs_lateral = max(self.maximum_abs_lateral, abs(self.last["lateral_deviation_m"]))
            self.minimum_along = min(self.minimum_along, self.last["along_track_progress_m"])
            self.maximum_along = max(self.maximum_along, self.last["along_track_progress_m"])
        if self.last["heading_drift_wrapped_rad"] is not None:
            self.maximum_abs_heading_wrapped = max(self.maximum_abs_heading_wrapped,
                                                   abs(self.last["heading_drift_wrapped_rad"]))
        return dict(self.last)

    def result(self) -> dict:
        wrapped = self.last["heading_drift_wrapped_rad"]
        unwrapped = self.last["heading_drift_unwrapped_rad"]
        max_wrapped = self.maximum_abs_heading_wrapped if self.valid_heading_samples else None
        max_unwrapped = self.maximum_abs_heading_unwrapped if self.unwrap_contiguous and self.samples else None
        lateral = self.last["lateral_deviation_m"]
        return {
            "sample_count": self.samples, "origin_world_position_m": list(self.origin),
            "initial_quaternion_xyzw": list(self.initial_quaternion),
            "initial_horizontal_forward_unit_xy": list(self.forward) if self.forward else None,
            "initial_horizontal_left_unit_xy": list(self.left) if self.left else None,
            "initial_heading_rad": self.initial_heading, "initial_axis_valid": self.forward is not None,
            "endpoint_world_position_m": list(self.previous_position),
            "along_track_progress_m": self.last["along_track_progress_m"],
            "minimum_along_track_position_m": self.minimum_along if self.forward else None,
            "maximum_along_track_position_m": self.maximum_along if self.forward else None,
            "endpoint_lateral_deviation_m": lateral,
            "endpoint_absolute_lateral_deviation_m": abs(lateral) if lateral is not None else None,
            "maximum_absolute_lateral_deviation_m": self.maximum_abs_lateral if self.forward else None,
            "horizontal_path_length_m": self.path_length,
            "endpoint_heading_drift_wrapped_rad": wrapped,
            "endpoint_heading_drift_wrapped_degrees": math.degrees(wrapped) if wrapped is not None else None,
            "maximum_absolute_heading_deviation_wrapped_rad": max_wrapped,
            "maximum_absolute_heading_deviation_wrapped_degrees": math.degrees(max_wrapped) if max_wrapped is not None else None,
            "endpoint_heading_drift_unwrapped_rad": unwrapped,
            "endpoint_heading_drift_unwrapped_degrees": math.degrees(unwrapped) if unwrapped is not None else None,
            "maximum_absolute_heading_deviation_unwrapped_rad": max_unwrapped,
            "maximum_absolute_heading_deviation_unwrapped_degrees": math.degrees(max_unwrapped) if max_unwrapped is not None else None,
            "valid_heading_physics_samples": self.valid_heading_samples,
            "invalid_heading_physics_samples": self.invalid_heading_samples,
            "heading_unwrap_contiguous_from_origin": self.unwrap_contiguous,
        }


def summarize_straightness(episodes) -> dict:
    paths = [episode["straightness"] for episode in episodes]
    axes_valid = bool(paths) and all(path["initial_axis_valid"] for path in paths)
    def maximum(name):
        values = [path[name] for path in paths if path[name] is not None]
        return max(values) if values else None
    return {
        "episode_count": len(paths), "all_initial_axes_valid": axes_valid,
        "sum_episode_along_track_progress_m": sum(path["along_track_progress_m"] for path in paths) if axes_valid else None,
        "maximum_absolute_lateral_deviation_m": maximum("maximum_absolute_lateral_deviation_m"),
        "maximum_absolute_endpoint_lateral_deviation_m": maximum("endpoint_absolute_lateral_deviation_m"),
        "maximum_absolute_heading_deviation_wrapped_rad": maximum("maximum_absolute_heading_deviation_wrapped_rad"),
        "all_heading_samples_valid": axes_valid and all(path["invalid_heading_physics_samples"] == 0 for path in paths),
        "all_reset_jumps_excluded": True,
    }


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return result


def positive_seconds(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("秒数必须为有限正数")
    return result


def seed_value(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("seed 必须为非负整数")
    return result


class VelocitySummary:
    """Plain arithmetic over measured samples, with no walking/pass threshold."""

    def __init__(self) -> None:
        self.samples = 0
        self.sums = [0.0] * 3
        self.error_sums = [0.0] * 3
        self.absolute_error_sums = [0.0] * 3
        self.squared_error_sums = [0.0] * 3
        self.max_absolute_errors = [0.0] * 3
        self.planar_error_sum = 0.0

    def add(self, measured, target) -> None:
        if not all(math.isfinite(float(value)) for value in (*measured, *target)):
            raise ValueError("物理速度/目标中有非有限值，不能生成有效误差指标")
        self.samples += 1
        errors = [float(value) - float(goal) for value, goal in zip(measured, target)]
        for index, (value, error) in enumerate(zip(measured, errors)):
            self.sums[index] += float(value)
            self.error_sums[index] += error
            self.absolute_error_sums[index] += abs(error)
            self.squared_error_sums[index] += error * error
            self.max_absolute_errors[index] = max(self.max_absolute_errors[index], abs(error))
        self.planar_error_sum += math.hypot(errors[0], errors[1])

    def result(self) -> dict:
        axes = {}
        for index, (axis, unit) in enumerate((("body_vx", "m/s"), ("body_vy", "m/s"), ("body_wz", "rad/s"))):
            axes[axis] = {
                "unit": unit,
                "measured_mean": self.sums[index] / self.samples if self.samples else None,
                "signed_error_mean": self.error_sums[index] / self.samples if self.samples else None,
                "mae": self.absolute_error_sums[index] / self.samples if self.samples else None,
                "rmse": math.sqrt(self.squared_error_sums[index] / self.samples) if self.samples else None,
                "max_absolute_error": self.max_absolute_errors[index] if self.samples else None,
            }
        return {"sample_count": self.samples, "axes": axes,
                "planar_velocity_error_norm_mean_mps": self.planar_error_sum / self.samples if self.samples else None}


def verify_config_contract(actual) -> dict:
    """Fail closed if the task no longer matches the audited observation contract."""
    from motrix_env_core.mdp.observations import BodyAngularVelocityObsCfg, BodyLinearVelocityObsCfg, CommandObsCfg
    from motrix_env_core.mdp.terminations import CollidingTerminationCfg

    value = actual.observations.value
    names = [field.name for field in fields(value)]
    if names != [name for name, _ in EXPECTED_TERMS]:
        raise ValueError(f"value 观测字段顺序与已审核合同不符: {names}")
    linear, angular = value.base_lin_vel, value.base_ang_vel
    if not isinstance(linear, BodyLinearVelocityObsCfg) or not isinstance(angular, BodyAngularVelocityObsCfg):
        raise ValueError("速度观测的实际 term 类型不符，不能猜测含义")
    if linear.body != "robot" or angular.body != "robot" or linear.scale != 2.0 or angular.scale != 0.25:
        raise ValueError("速度观测的对象/缩放与已审核合同不符")
    noises = {}
    for field in fields(value):
        term = getattr(value, field.name)
        if hasattr(term, "noise"):
            noises[field.name] = float(term.noise.amplitude)
            if noises[field.name] != 0.0:
                raise ValueError(f"value.{field.name} 含观测噪声，不能作为无噪声物理交叉校验")
    if not isinstance(value.command, CommandObsCfg) or value.command.command_name != "walk":
        raise ValueError("value.command 不是已审核的 walk 指令")
    terminations = actual.termination_cfgs()
    if set(terminations) != {"colliding"} or not isinstance(terminations["colliding"], CollidingTerminationCfg):
        raise ValueError(f"实际 termination 并非唯一 colliding，不能把 aggregated terminated 归因为躯干触地: {list(terminations)}")
    collision = terminations["colliding"]
    if not collision.termination_geoms or not collision.ground_geom:
        raise ValueError("碰撞终止没有有效几何体/地面配置")
    return {
        "value_terms": [{"name": name, "size": size} for name, size in EXPECTED_TERMS],
        "linear_velocity_slice": [0, 3], "linear_velocity_scale": float(linear.scale),
        "angular_velocity_slice": [3, 6], "angular_velocity_scale": float(angular.scale),
        "command_slice": [9, 12], "value_noise_amplitudes": noises,
        "raw_critic_before_agent_normalizer": True,
        "velocity_frame": "robot base/body frame; same rotation as official tracking rewards",
        "angular_z_note": "body_wz 与任务奖励一致；身体倾斜时不等于欧拉 yaw 角的时间导数",
        "termination_geoms": list(collision.termination_geoms), "ground_geom": collision.ground_geom,
        "termination_attribution": "terminated == colliding, verified sole configured termination term",
        "max_episode_seconds": float(actual.max_episode_seconds),
        "control_dt_seconds": float(actual.ctrl_dt), "physics_dt_seconds": float(actual.sim.dt),
        "physical_randomization_enabled": bool(actual.sim_reset.humanoid_state.randomization.enabled),
        "policy_joint_observation_noise_is_unchanged": True,
    }


def verify_runtime_layout(env) -> dict:
    sizes = {"policy": env.policy_observation_space.shape[-1],
             "value": env.value_observation_space.shape[-1], "action": env.action_space.shape[-1]}
    if sizes != {"policy": 55, "value": 58, "action": 14}:
        raise ValueError(f"实际空间不符合55/58/14合同: {sizes}")
    entries = env.observation_groups["value"].terms
    observed = [(entry.name, entry.size) for entry in entries]
    if observed != EXPECTED_TERMS:
        raise ValueError(f"真实运行时 value terms 与已审核合同不符: {observed}")
    return {"actual_spaces": sizes, "runtime_value_terms": [{"name": name, "size": size} for name, size in observed]}


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实 MotrixSim/FastSAC CPU 行走评估；逐步采样、排除auto-reset污染，不自动判定走稳。")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-dir", required=True, help="含metadata.json和task_config.yaml的真实FastSAC训练目录")
    parser.add_argument("--checkpoint", required=True, help="run_dir内checkpoint相对路径，或其绝对路径")
    parser.add_argument("--settings", required=True, help="该次训练保存的course_config.json")
    parser.add_argument("--seconds", type=positive_seconds, default=20.0, help="模拟秒数，默认20；不是墙钟超时")
    parser.add_argument("--seed", type=seed_value, default=42)
    parser.add_argument("--output", required=True, help="尚不存在的新输出目录")
    parser.add_argument("--inspect", action="store_true", help="仅验证配置/真实checkpoint，不创建仿真环境、不运行策略")
    return parser


def evaluate(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    started = time.monotonic()
    report = {
        "schema": 2, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "simulator": "motrixsim", "policy_inference_device": "cpu", "rendering_enabled": False,
        "requested_simulated_seconds": args.seconds, "seed": args.seed,
        "training_executed": False, "walking_verified": False, "automatic_acceptance_decision": None,
        "velocity_metrics_computed": False, "straightness_metrics_computed": False, "completed_control_steps": 0,
        "limitations": [
            "躯干触地终止统计仅覆盖配置列出的几何体接触地面，不代表所有步态失败。",
            "timeout 与躯干触地分开统计；auto-reset 不代表策略自行恢复。",
            "未终止不能证明向前行走、速度跟踪、姿态稳定或迁移成功。",
            "body_wz 是机身坐标系角速度z分量，身体倾斜时不是欧拉yaw变化率。",
            "直线性固定使用每回合真实初始局部+X的世界XY投影；不使用最终位移方向或拟合直线。",
            "朝向偏差由四元数水平投影直接计算，不积分body_wz；投影退化时明确输出无效。",
            "unwrapped朝向假设相邻有效采样的真实水平转角小于π；任何退化采样后不跨缺口展开。",
            "保留官方观测噪声及物理随机化；单一seed/有限时长不能代表全部工况。",
        ],
    }
    write_json(report_path, report)
    wrapper, raw_env, original_physics_step = None, None, None
    try:
        root = Path(args.root).resolve()
        sys.path.insert(0, str(root / "src"))
        from microduck_course.config import COURSE_TASK, UPSTREAM_COMMIT, evaluation_settings, resolve_path
        from microduck_course.runtime import checkpoint_state, setup_runtime
        from microduck_course.storage import policy_provenance, sha256, validate_policy_files

        run_dir = resolve_path(root, args.run_dir)
        checkpoint = resolve_path(run_dir, args.checkpoint)
        checkpoint.relative_to(run_dir)
        settings_source = resolve_path(root, args.settings)
        source_hashes = {"checkpoint": sha256(checkpoint), "metadata.json": sha256(run_dir / "metadata.json"),
                         "task_config.yaml": sha256(run_dir / "task_config.yaml"), "settings": sha256(settings_source)}
        metadata = validate_policy_files(run_dir, checkpoint, source_hashes["checkpoint"])
        if metadata.get("train_backend") != "torch":
            raise ValueError("此工具只评估已审核的官方FastSAC Torch策略")
        state = checkpoint_state(checkpoint)
        report.update(source=policy_provenance(run_dir, checkpoint, "walking_evaluation"),
                      global_step=state["global_step"], update_idx=state["update_idx"], source_file_hashes=source_hashes)
        settings = evaluation_settings(read_json(settings_source))
        if settings["course"]["upstream_commit"] != UPSTREAM_COMMIT or settings["course"]["simulator"] != "motrixsim":
            raise ValueError("保存设置没有匹配本课程固定MotrixLab/MotrixSim版本")
        settings["evaluation"]["seed"] = args.seed
        settings["runtime"]["device"] = "cpu"
        target = tuple(float(value) for value in settings["evaluation"]["command"])
        if len(target) != 3 or not all(math.isfinite(value) for value in target):
            raise ValueError("evaluation.command 必须为三个有限的vx/vy/body_wz目标")
        snapshot = output / "evaluation_settings.json"
        write_json(snapshot, settings)
        snapshot_hash = sha256(snapshot)
        report.update(evaluation=settings["evaluation"], evaluation_settings_sha256=snapshot_hash,
                      policy_and_simulation_executed=False)
        setup_runtime(root, settings, snapshot)
        from motrix_env_core import registry
        from motrix_rl import frameworks, runs

        actual = registry.make_env_config(COURSE_TASK, mode="play")
        contract = verify_config_contract(actual)
        if actual.commands.walk.stand_prob != 0.0 or any(tuple(bounds) != target for bounds in actual.commands.walk.vel_limit):
            raise ValueError("实际评估命令分布未固定为请求值，拒绝计算不对应的目标误差")
        report["contract"] = contract
        provider = frameworks.get_agent_provider(metadata["rllib"], metadata["algo"], metadata["train_backend"])
        if provider is None:
            raise ValueError("没有找到真实官方FastSAC provider")
        source_config = runs.read_task_config(run_dir, provider.config_type)
        if args.inspect:
            report["status"] = "inspected_without_simulation"
            report["actual_spaces"] = None
            report["inspection_note"] = "仅检查真实checkpoint、配置term顺序/类型/scale/noise；运行时55/58/14将在实际评估中确认。"
            return 0

        import numpy as np
        import torch
        from motrix_env_core.numba.math.quaternion import rotate_inverse_components
        from motrix_env_core.sim import LinkAngularVelocityQuery, LinkLinearVelocityQuery, LinkPositionQuery, LinkQuaternionQuery
        from motrix_rl_builtin.fastsac.agent import FastSacAgent
        from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = torch.device("cpu")
        raw_env = registry.make(COURSE_TASK, num_envs=1, mode="play", sim="motrixsim", seed=args.seed)
        wrapper = FastSacNpEnvWrap(raw_env, device, render=None)
        obs, critic_obs = wrapper.reset()
        report.update(verify_runtime_layout(raw_env))
        if tuple(obs.shape) != (1, 55) or tuple(critic_obs.shape) != (1, 58):
            raise ValueError("实际初始化观测不是单环境55/58维")
        if not torch.isfinite(obs).all() or not torch.isfinite(critic_obs).all():
            raise ValueError("初始化观测含非有限值")
        agent = FastSacAgent(
            obs_dim=55, critic_obs_dim=58, act_dim=14, num_envs=1,
            cfg=source_config.algo.agent, device=device,
            action_scale=(wrapper.action_high - wrapper.action_low) / 2.0,
            action_bias=(wrapper.action_high + wrapper.action_low) / 2.0,
        )
        agent.load_state_dict(state, load_optimizers=False)
        del state
        agent.actor.eval()
        agent.obs_normalizer.eval()
        agent.critic_obs_normalizer.eval()
        if any(parameter.device.type != "cpu" for parameter in agent.actor.parameters()):
            raise ValueError("策略参数未全部位于CPU")

        base = raw_env.cfg.scene.objs.robot.resolved_base_link_name
        physics_reads = raw_env.sim.compile_reads({
            "position": LinkPositionQuery(link=base),
            "quaternion": LinkQuaternionQuery(link=base),
            "linear_world": LinkLinearVelocityQuery(link=base),
            "angular_world": LinkAngularVelocityQuery(link=base),
        })
        latest = {}
        sample_counter = 0
        original_physics_step = raw_env.physics_step

        def read_native() -> dict:
            # Copies are essential: execute() refreshes the shared query views.
            physics_reads.execute()
            measured = {}
            for name, dimension in (("position", 3), ("quaternion", 4), ("linear_world", 3), ("angular_world", 3)):
                array = np.asarray(physics_reads.view(name)).copy()
                if array.shape != (1, dimension) or not np.isfinite(array).all():
                    raise ValueError(f"真实物理查询{name}形状/数值异常: {array.shape}")
                measured[name] = array[0]
            return measured

        def sample_after_physics() -> None:
            nonlocal sample_counter
            original_physics_step()
            # Read-only native queries: no extra step, reset, observe or RNG consumption.
            latest.update(read_native())
            sample_counter += 1

        initial_native = read_native()
        straightness = StraightnessSummary(initial_native["position"], initial_native["quaternion"])
        reset_pose_reads = 1
        raw_env.physics_step = sample_after_physics
        control_dt = float(raw_env.cfg.ctrl_dt)
        requested_steps = max(1, math.ceil(args.seconds / control_dt - 1e-10))
        report.update(policy_and_simulation_executed=True, requested_control_steps=requested_steps,
                      sampling_point="immediately_after_actual_physics_step_before_transition_and_automatic_reset",
                      velocity_source="public SimBackend.compile_reads q/world_v/world_w, rotated with official rotate_inverse_components",
                      position_source="public SimBackend.compile_reads LinkPositionQuery; actual base world XYZ",
                      heading_source="public LinkQuaternionQuery xyzw; normalized quaternion rotates local +X into world XY",
                      heading_projection_norm_epsilon=HEADING_PROJECTION_EPSILON,
                      straightness_reference="each actual reset position and initial horizontal local +X; fixed within episode",
                      lateral_sign="positive to the left of the episode's initial horizontal forward direction",
                      heading_sign="positive counterclockwise in world XY; wrapped range [-pi, pi]",
                      horizontal_path_length_definition="sum of successive world XY distances within each episode, including origin-to-first-step and terminal physics",
                      extra_physics_steps_for_measurement=0, extra_observation_calls_for_measurement=0)
        all_metrics, nonterminal_metrics = VelocitySummary(), VelocitySummary()
        episode_index, episode_steps, current_nonterminal_steps, longest_nonterminal_steps = 0, 0, 0, 0
        collisions, timeouts, resets, crosschecked = 0, 0, 0, 0
        maximum_crosscheck_difference = 0.0
        episodes = []
        csv_path = output / "steps.csv"
        trajectory_path = output / "trajectory.csv"
        trajectory_rows = 0

        def trajectory_row(native, metrics, kind, step, elapsed, terminated=False, truncated=False):
            row = {
                "sample_kind": kind, "control_step": step, "simulated_seconds": step * control_dt,
                "episode_index": episode_index, "episode_elapsed_seconds": elapsed,
                "terminated_collision": terminated, "truncated_timeout": truncated,
                "automatic_reset_occurred": terminated or truncated,
            }
            row.update({f"world_{axis}_m": float(value) for axis, value in zip("xyz", native["position"])})
            row.update({f"base_q{axis}": float(value) for axis, value in zip("xyzw", native["quaternion"])})
            row.update(metrics)
            return row

        with (csv_path.open("w", encoding="utf-8-sig", newline="") as stream,
              trajectory_path.open("w", encoding="utf-8-sig", newline="") as trajectory_stream):
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            trajectory_writer = csv.DictWriter(trajectory_stream, fieldnames=TRAJECTORY_FIELDS)
            trajectory_writer.writeheader()
            trajectory_writer.writerow(trajectory_row(initial_native, straightness.origin_point, "episode_origin", 0, 0.0))
            trajectory_rows += 1
            with torch.inference_mode():
                for step in range(1, requested_steps + 1):
                    previous_samples = sample_counter
                    actions = agent.act(obs, deterministic=True)
                    if tuple(actions.shape) != (1, 14) or not torch.isfinite(actions).all():
                        raise ValueError("实际策略输出不是有限的1×14动作")
                    result = wrapper.step(actions)
                    if sample_counter != previous_samples + 1:
                        raise ValueError("一个control step未恰好对应一个真实物理采样")
                    terminated, truncated = bool(result.terminated[0].item()), bool(result.truncated[0].item())
                    done = terminated or truncated
                    q, world_v, world_w = latest["quaternion"], latest["linear_world"], latest["angular_world"]
                    if not math.isclose(float(np.dot(q, q)), 1.0, abs_tol=1e-3):
                        raise ValueError("实际基座四元数未单位化，拒绝猜测body速度")
                    body_v = tuple(float(value) for value in rotate_inverse_components(q, world_v))
                    body_w = tuple(float(value) for value in rotate_inverse_components(q, world_w))
                    measured = (body_v[0], body_v[1], body_w[2])
                    all_metrics.add(measured, target)
                    if not done:
                        raw_critic = result.critic_obs.detach().cpu().numpy()[0].copy()
                        decoded = np.concatenate((raw_critic[0:3] / 2.0, raw_critic[3:6] / 0.25))
                        native = np.asarray(body_v + body_w)
                        difference = float(np.max(np.abs(decoded - native)))
                        maximum_crosscheck_difference = max(maximum_crosscheck_difference, difference)
                        if not np.allclose(decoded, native, atol=2e-4, rtol=2e-4):
                            raise ValueError(f"原始critic还原速度与reset前原生查询不符(max_difference={difference})")
                        if not np.allclose(raw_critic[9:12], target, atol=1e-6, rtol=0):
                            raise ValueError("实际原始critic中的固定指令与评估目标不符")
                        crosschecked += 1
                        nonterminal_metrics.add(measured, target)
                    reward = float(result.reward[0].item())
                    if not math.isfinite(reward):
                        raise ValueError("实际transition reward为非有限值")
                    episode_steps += 1
                    path_point = straightness.add(latest["position"], q)
                    collisions += int(terminated)
                    timeouts += int(truncated)
                    resets += int(done)
                    current_nonterminal_steps = 0 if done else current_nonterminal_steps + 1
                    longest_nonterminal_steps = max(longest_nonterminal_steps, current_nonterminal_steps)
                    errors = tuple(value - goal for value, goal in zip(measured, target))
                    row = {
                        "control_step": step, "simulated_seconds": step * control_dt,
                        "episode_index": episode_index, "episode_elapsed_seconds": episode_steps * control_dt,
                        "terminated_collision": terminated, "truncated_timeout": truncated,
                        "automatic_reset_occurred": done, "post_reset_returned_observation_excluded": done,
                        "raw_critic_crosscheck_used": not done,
                        "target_vx_mps": target[0], "target_vy_mps": target[1], "target_body_wz_radps": target[2],
                        "vx_error_mps": errors[0], "vy_error_mps": errors[1], "body_wz_error_radps": errors[2],
                        "planar_velocity_error_mps": math.hypot(errors[0], errors[1]), "transition_reward": reward,
                    }
                    for name, values in (("body_v", body_v), ("body_w", body_w), ("world_v", world_v), ("world_w", world_w)):
                        unit = "mps" if name.endswith("v") else "radps"
                        for axis, value in zip("xyz", values):
                            row[f"{name}{axis}_{unit}"] = float(value)
                    row.update({f"base_q{axis}": float(value) for axis, value in zip("xyzw", q)})
                    writer.writerow(row)
                    trajectory_writer.writerow(trajectory_row(latest, path_point, "pre_reset_physics", step,
                                                               episode_steps * control_dt, terminated, truncated))
                    trajectory_rows += 1
                    report["completed_control_steps"] = step
                    obs = result.obs
                    if done:
                        episodes.append({"episode_index": episode_index, "control_steps": episode_steps,
                                         "simulated_seconds": episode_steps * control_dt, "end_step": step,
                                         "collision_termination": terminated, "timeout": truncated,
                                         "censored_at_evaluation_end": False,
                                         "straightness": straightness.result()})
                        episode_index += 1
                        episode_steps = 0
                        if step < requested_steps:
                            # wrapper.step has reset this row; this pose starts a NEW
                            # segment. Never overwrite the terminal sample in latest.
                            reset_native = read_native()
                            reset_pose_reads += 1
                            straightness = StraightnessSummary(reset_native["position"], reset_native["quaternion"])
                            trajectory_writer.writerow(trajectory_row(reset_native, straightness.origin_point,
                                                                       "episode_origin", step, 0.0))
                            trajectory_rows += 1
                    if step % 100 == 0:
                        stream.flush()
                        trajectory_stream.flush()
        if episode_steps:
            episodes.append({"episode_index": episode_index, "control_steps": episode_steps,
                             "simulated_seconds": episode_steps * control_dt, "end_step": requested_steps,
                             "collision_termination": False, "timeout": False,
                             "censored_at_evaluation_end": True,
                             "straightness": straightness.result()})
        if agent.global_step != report["global_step"] or agent.update_idx != report["update_idx"]:
            raise ValueError("评估期间训练计数改变，不能标记只读策略评估成功")
        current_hashes = {"checkpoint": sha256(checkpoint), "metadata.json": sha256(run_dir / "metadata.json"),
                          "task_config.yaml": sha256(run_dir / "task_config.yaml"), "settings": sha256(settings_source)}
        if current_hashes != source_hashes or sha256(snapshot) != snapshot_hash:
            raise ValueError("源策略/配置或评估快照在运行期间变化")
        report.update(
            status="completed", velocity_metrics_computed=True, straightness_metrics_computed=True,
            actual_simulated_seconds=requested_steps * control_dt,
            collision_termination_count=collisions, timeout_count=timeouts, automatic_reset_count=resets,
            longest_contiguous_nonterminal_seconds=longest_nonterminal_steps * control_dt,
            longest_contiguous_nonterminal_definition="连续既无terminated也无truncated的完整control steps×ctrl_dt；边界步保守排除，reset后另起段",
            longest_episode_duration_seconds=max((item["simulated_seconds"] for item in episodes), default=0.0),
            episodes=episodes, all_physics_samples=all_metrics.result(),
            straightness=summarize_straightness(episodes),
            nonterminal_physics_samples=nonterminal_metrics.result(),
            raw_critic_crosschecked_sample_count=crosschecked,
            post_auto_reset_critic_observations_excluded=resets,
            maximum_raw_critic_vs_native_velocity_difference=maximum_crosscheck_difference if crosschecked else None,
            source_hashes_unchanged=True,
            csv={"path": str(csv_path), "rows": requested_steps, "sha256": sha256(csv_path),
                 "all_rows_use_pre_reset_physics": True},
            trajectory_csv={"path": str(trajectory_path), "rows": trajectory_rows, "sha256": sha256(trajectory_path),
                            "physics_rows": requested_steps, "episode_origin_rows": reset_pose_reads,
                            "terminal_rows_are_pre_reset_physics": True,
                            "origin_rows_are_actual_reset_poses": True, "segments_joined_across_reset": False},
            read_only_reset_pose_queries=reset_pose_reads,
        )
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if raw_env is not None and original_physics_step is not None:
            raw_env.physics_step = original_physics_step
        if wrapper is not None:
            wrapper.close()  # This official wrapper has close(); the raw env does not.
        report["wall_clock_seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path),
                          "completed_control_steps": report["completed_control_steps"],
                          "walking_verified": False}, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = arguments().parse_args(argv)
    try:
        return evaluate(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    for output_stream in (sys.stdout, sys.stderr):
        if hasattr(output_stream, "reconfigure"):
            output_stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
