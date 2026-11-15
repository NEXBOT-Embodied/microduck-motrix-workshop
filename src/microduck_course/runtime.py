"""Thin adapter to real MotrixLab APIs; no RL algorithm is reimplemented here."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, fields, replace
from pathlib import Path

from microduck_course.config import COURSE_TASK, CourseError, resolved_algorithm, resolve_path, training_visualization_options
from microduck_course.storage import sha256, validate_export_selection, write_json

RUNTIME_MODULES = ("motrixsim", "torch", "hydra", "omegaconf", "motrix_env_core", "motrix_env_motrixsim", "motrix_robots", "motrix_envs", "motrix_rl", "motrix_rl_builtin")


def add_upstream_sources(root: Path, settings: dict) -> Path:
    upstream = resolve_path(root, settings["course"]["upstream_path"])
    if not (upstream / "motrix_rl" / "src" / "motrix_rl" / "runner.py").is_file():
        raise CourseError(f"缺少固定版 MotrixLab 源码: {upstream}")
    for path in sorted(upstream.glob("*/src")):
        source = str(path)
        if source not in sys.path:
            sys.path.insert(0, source)
    return upstream


def setup_runtime(root: Path, settings: dict, snapshot: Path) -> None:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    os.environ.setdefault("OMP_NUM_THREADS", str(settings["runtime"]["cpu_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(settings["runtime"]["cpu_threads"]))
    os.environ["MICRODUCK_COURSE_ACTIVE_CONFIG"] = str(snapshot.resolve())
    add_upstream_sources(root, settings)
    missing = []
    for name in RUNTIME_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        raise CourseError("镜像运行依赖未就绪（请在课前构建，不在现场安装）：\n" + "\n".join(missing))
    import torch
    from microduck_course.task import register
    torch.set_num_threads(settings["runtime"]["cpu_threads"])
    register()
    from microduck_course.compat import install_render_compatibility
    from motrix_env_core import registry
    install_render_compatibility(registry.make_env_config(COURSE_TASK).scene.objs.robot.resolved_base_link_name)


def checkpoint_state(path: Path, *, require_optimizers: bool = False) -> dict:
    import torch
    # policies.yaml only accepts teacher-verified local artifacts with recorded hashes.
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {"actor", "qnet", "qnet_target", "log_alpha", "obs_normalizer", "critic_obs_normalizer", "global_step", "update_idx"}
    if require_optimizers:
        required.update(("q_optimizer", "actor_optimizer", "alpha_optimizer"))
    if not isinstance(data, dict) or required - data.keys():
        raise CourseError(f"checkpoint 不包含本课程所需的 FastSAC 状态: {path}")
    for key in ("global_step", "update_idx"):
        if isinstance(data[key], bool) or not isinstance(data[key], int) or data[key] < 0:
            raise CourseError(f"checkpoint {key} 无效。")
    return data


def actor_update_gates(initial_update: int, final_update: int, frequency: int) -> int:
    """Count upstream FastSAC actor gates in [initial_update, final_update)."""
    return (final_update - 1) // frequency - (initial_update - 1) // frequency


def require_learning_progress(initial_update: int, final_update: int, frequency: int) -> dict:
    critic_updates = final_update - initial_update
    gates = actor_update_gates(initial_update, final_update, frequency)
    if critic_updates <= 0 or gates <= 0:
        raise CourseError("本次仅采样或未跨过 actor 更新门；不能标记完成训练。")
    return {"initial_update_idx": initial_update, "final_update_idx": final_update, "new_critic_updates": critic_updates, "new_actor_update_gates": gates}


def build_train_config(root: Path, settings: dict, starter_run: Path | None, starter_checkpoint: Path | None):
    from omegaconf import OmegaConf
    from motrix_rl.config import CheckpointConfig, LoggingConfig, TaskMeta, TrainConfig
    from motrix_rl_builtin.fastsac.config import FastSacCfg

    algo = resolved_algorithm(root, settings, starter_run)
    resumed_step = 0
    if starter_run is not None:
        resumed_step = checkpoint_state(starter_checkpoint, require_optimizers=True)["global_step"]
    # Official FastSAC interprets this as an absolute target. Add iterations to avoid a no-op resume.
    algo["trainer"]["num_learning_iterations"] = resumed_step + settings["runtime"]["additional_iterations"]
    typed_algo = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(FastSacCfg), algo))
    config = TrainConfig(
        task=TaskMeta(env=COURSE_TASK, rllib="motrix", algo="fastsac", train_backend="torch"),
        num_envs=settings["runtime"]["num_envs"], play_num_envs=settings["runtime"]["play_num_envs"],
        seed=settings["runtime"]["seed"],
        logging=LoggingConfig(backend="tensorboard", interval=settings["runtime"]["logging_interval"]),
        checkpoint=CheckpointConfig(interval=settings["runtime"]["checkpoint_interval"]),
        algo=typed_algo, sim="motrixsim", resume=str(starter_checkpoint) if starter_checkpoint else None,
        render=training_visualization_options(settings.get("training_visualization", False))["enabled"], play=False,
    )
    return config, resumed_step


def train_real(root: Path, settings: dict, path: Path, starter_run: Path | None, starter_checkpoint: Path | None) -> dict:
    setup_runtime(root, settings, path / "course_config.json")
    from motrix_rl import runner
    from motrix_env_core import registry

    config, initial_step = build_train_config(root, settings, starter_run, starter_checkpoint)
    initial_update = checkpoint_state(starter_checkpoint, require_optimizers=True)["update_idx"] if starter_checkpoint else 0
    agent_cfg = config.algo.agent
    for name, minimum in (("learning_starts", 0), ("num_updates", 1), ("policy_frequency", 1)):
        value = getattr(agent_cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise CourseError(f"FastSAC {name} 必须是 >= {minimum} 的整数。")
    expected_updates = max(0, settings["runtime"]["additional_iterations"] - agent_cfg.learning_starts) * agent_cfg.num_updates
    require_learning_progress(initial_update, initial_update + expected_updates, agent_cfg.policy_frequency)
    actual_task = registry.make_env_config(COURSE_TASK)
    write_json(path / "task_effective.json", {
        "status": "task_factory_instantiated_before_simulation",
        "env_name": COURSE_TASK, "simulator": "motrixsim",
        "sim_dt": actual_task.sim.dt, "ctrl_dt": actual_task.ctrl_dt,
        "reward_weights": {field.name: getattr(actual_task.rewards, field.name).weight for field in fields(actual_task.rewards)},
        "reward_terms": {field.name: asdict(getattr(actual_task.rewards, field.name)) for field in fields(actual_task.rewards)},
        "walk_command": asdict(actual_task.commands.walk),
        "model_file": str(actual_task.scene.objs.robot.model.file),
        "joint_order": list(actual_task.scene.objs.robot.key_pose.joint_names),
        "walking_constraints": settings.get("walking_constraints"),
    })
    from microduck_course.training_view import TrainingView
    visual_options = training_visualization_options(settings.get("training_visualization", False))
    with TrainingView(visual_options, path, initial_step, starter_checkpoint) as view:
        result = runner.train(runner.TrainRequest(config=config, runs_root=path / "motrix_runs", render=view.render_config))
    policy = result.find_best_policy().resolve()
    if not policy.is_file() or policy.stat().st_size == 0:
        raise CourseError("训练返回但没有非空 checkpoint；不标记成功。")
    final_state = checkpoint_state(policy)
    final_step = final_state["global_step"]
    target = initial_step + settings["runtime"]["additional_iterations"]
    if final_step < target or final_step <= initial_step:
        raise CourseError(f"没有完成新增采样/训练：initial={initial_step}, final={final_step}, target={target}")
    learning = require_learning_progress(initial_update, final_state["update_idx"], agent_cfg.policy_frequency)
    return {
        "upstream_run_dir": result.run_dir.resolve().relative_to(path.resolve()).as_posix(),
        "checkpoint": policy.relative_to(result.run_dir.resolve()).as_posix(),
        "checkpoint_sha256": sha256(policy), "initial_global_step": initial_step,
        "final_global_step": final_step, "additional_iterations": final_step - initial_step,
        "new_environment_transitions": (final_step - initial_step) * settings["runtime"]["num_envs"],
        "walking_verified": False, **learning,
        **({"training_visualization": view.report} if view.enabled else {}),
    }


def play_real(root: Path, settings: dict, snapshot: Path, run_dir: Path, policy: Path, *, video: Path | None = None, seconds: int = 20) -> dict:
    setup_runtime(root, settings, snapshot)
    from motrix_env_core.renderer import RenderConfig
    from motrix_env_core import registry
    from motrix_rl import frameworks, runs
    run = runs.open_run_context(run_dir)
    provider = frameworks.get_agent_provider(run.metadata.rllib, run.metadata.algo, run.metadata.train_backend)
    if provider is None:
        raise CourseError("找不到源策略对应的官方 FastSAC provider。")
    original_config = runs.read_task_config(run_dir, provider.config_type)
    # Use the public inference context API so original-task reference policies
    # receive exactly the same course evaluation factory as student policies.
    # The original metadata/task_config files remain untouched.
    evaluation_run = replace(run, metadata=replace(run.metadata, env_name=COURSE_TASK, sim="motrixsim"))
    actual = registry.make_env_config(COURSE_TASK)
    render = RenderConfig(headless=True, path=video, fps=30, num_frames=seconds * 30, width=960, height=540) if video else RenderConfig()
    # One robot needs very little policy compute. Use CPU for evaluation so
    # recording can coexist with teacher CUDA training; MotrixSim still owns
    # physics and graphics. All compared stages use this same device setting.
    playback_algo = replace(original_config.algo, device="cpu")
    context = frameworks.create_trainer_context(
        evaluation_run, num_envs=original_config.num_envs, play_num_envs=1,
        seed=settings["evaluation"]["seed"], rl_cfg=playback_algo,
        logging=original_config.logging, checkpoint=original_config.checkpoint, render=render,
    )
    handle = frameworks.create_trainer(context)
    handle.play(str(policy))
    from microduck_course.compat import APPLIED_ADAPTERS, PRESENTATION, RENDER_EVENTS
    return {
        "status": "playback_returned", "simulator": "motrixsim",
        "render_compatibility_adapters": sorted(APPLIED_ADAPTERS),
        "render_presentation": PRESENTATION,
        "render_events": list(RENDER_EVENTS),
        "policy_inference_device": "cpu",
        "source_env": run.metadata.env_name, "evaluation_env": COURSE_TASK,
        "evaluation": settings["evaluation"],
        "actual_command_bounds": [list(row) for row in actual.commands.walk.vel_limit],
        "actual_stand_probability": actual.commands.walk.stand_prob,
        "walking_verified": False, "velocity_metrics_computed": False,
        "limitations": ["0.15 m/s 是待验收试配命令，不是实测速度", "环境噪声和物理随机化仍依官方任务设置", "需要教师观察真实回放，不自动判定行走成功或速度误差"],
    }


def export_real(root: Path, settings: dict, snapshot: Path, run_dir: Path, policy: Path, expected_hash: str, output: Path) -> dict:
    setup_runtime(root, settings, snapshot)
    from motrix_rl import checkpoints, runs
    from motrix_rl.deploy import export_onnx
    run = runs.open_run_context(run_dir)
    validate_export_selection(checkpoints.best_policy(run.metadata, run_dir), policy, expected_hash)
    result = export_onnx(run_dir, output, validation_seed=settings["runtime"]["seed"], validation_samples=32)
    try:
        validate_export_selection(checkpoints.best_policy(run.metadata, run_dir), policy, expected_hash)
    except CourseError:
        result.path.unlink(missing_ok=True)
        raise
    report = {"status": "onnx_parity_passed", "path": str(result.path), "sha256": sha256(result.path), "source_checkpoint": str(policy), "source_checkpoint_sha256": expected_hash, "report": asdict(result.report), "sim2sim_completed": False}
    write_json(output.with_suffix(".report.json"), report)
    return report


def runtime_probe(root: Path, settings: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="microduck-runtime-") as temporary:
        snapshot = Path(temporary) / "course_config.json"
        write_json(snapshot, settings)
        setup_runtime(root, settings, snapshot)
        from motrix_env_core import registry
        from motrix_envs.locomotion.humanoid.microduck import make_microduck_walk_flat_cfg
        base = make_microduck_walk_flat_cfg()
        actual = registry.make_env_config(COURSE_TASK)
        expected = (settings["walking_constraints"]["task"]["rewards"]["penalty_action_rate"]["weight"]
                    if "walking_constraints" in settings else
                    base.rewards.penalty_action_rate.weight * settings["experiment"]["action_rate_multiplier"])
        if actual.rewards.penalty_action_rate.weight != expected:
            raise CourseError("实验配置没有注入实际任务 reward。")
        env = registry.make(COURSE_TASK, num_envs=1, sim="motrixsim", seed=settings["runtime"]["seed"])
        try:
            # registry.make returns the raw manager environment, whose reset
            # requires explicit env_ids. init_state performs its full first reset.
            state = env.init_state()
            sizes = {"policy": env.policy_observation_space.shape[-1], "value": env.value_observation_space.shape[-1], "action": env.action_space.shape[-1]}
            if sizes != {"policy": 55, "value": 58, "action": 14}:
                raise CourseError(f"实测策略契约与静态推导不一致: {sizes}")
            import numpy as np
            observed = {"policy": list(state.obs.policy.shape), "value": list(state.obs.value_or_policy.shape)}
            if observed != {"policy": [1, 55], "value": [1, 58]} or not all(np.isfinite(x).all() for x in (state.obs.policy, state.obs.value_or_policy)):
                raise CourseError(f"实际 reset 观测形状或数值无效: {observed}")
            return {"status": "runtime_reset_passed", "actual_spaces": sizes, "actual_reset_observations": observed, "reset_observations_finite": True, "actual_action_rate_weight": expected, "walking_verified": False, "render_verified": False}
        finally:
            # The raw ArrayEnv has no close API in this pinned upstream version.
            close = getattr(env, "close", None)
            if callable(close):
                close()
