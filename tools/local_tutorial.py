"""Small local entry point for the two audited MotrixSim teaching checkpoints.

Examples (use the prepared runtime Python, from the course directory)::
    python tools/local_tutorial.py check
    python tools/local_tutorial.py play --case straight
    python tools/local_tutorial.py train --steps 256 --yaw-weight -25 --render --dry-run

This file delegates simulation, playback and learning to the existing course
implementation. It never registers these unverified policies as classroom ready.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microduck_course.config import CourseError, read_yaml, resolved_algorithm, training_visualization_options
from microduck_course.storage import course_run_policy, new_run, policy_provenance, sha256, validate_policy_files, write_json

PYTHON = Path(sys.executable)
# 原始模型与配置只读保存；课堂产生的新结果写在工程自己的 outputs 里。
SOURCE_RUNS = ROOT / "policies/source_runs"
TRAIN_OUTPUTS = ROOT / "outputs/training_outputs"
PLAY_OUTPUTS = ROOT / "outputs/local_tutorial"
TRAINING_GROUP = TRAIN_OUTPUTS / "local_tutorial"
LAST_TRAINING = TRAINING_GROUP / "last_completed_training.json"
PROFILES = {
    -15.0: "forward_heading_rate_v1",
    -20.0: "forward_heading_rate_mid_v1",
    -25.0: "forward_heading_rate_strong_v1",
}
CASES = {
    "drift": {
        "course_run": "teacher_forward_pilot/20261108T151558Z-45791b71",
        "upstream_run": "26-11-08_23-15-59-615831",
        "checkpoint": "checkpoints/latest.pt",
        "sha256": "69510a7fd132411319a7c6ac3b538f3d452af7201f8de51e2b2fa3185dc28f02",
        "course_config_sha256": "5afc39e14e500b80397936a7caaaa3f805854d76aec9bea3fc5800836ea44707",
        "metadata_sha256": "d7ae4861807691654d6edbf9daa8d1326eda7d0b98687ce1b4f62737acda5b93",
        "task_config_sha256": "f0e87d8b1c33d32aa439e3f893545ae23a5a5575ef89f0725471f2e33df0069a",
        "global_step": 18000,
        "update_idx": 65680,
        "label": "走偏示例：用于观察方向偏移；原训练作业已完成",
    },
    "straight": {
        "course_run": "teacher_heading_continue28/20261109T085225Z-e8b51aba",
        "upstream_run": "26-11-09_16-52-29-008179",
        "checkpoint": "checkpoints/model_0026000.pt",
        "sha256": "a316ab0a9f42336e0d445c55b6aee35ef93427276b311c4b51a00234f41f4ca8",
        "course_config_sha256": "cf706f4cbf96702c38ae02083a5c0252d69b6f20c6766a78d90d5903d0abfcf2",
        "metadata_sha256": "63d1fa65bf76d2589d7387add203ac885eef5f7ee32a54ef332e251fdc4948dc",
        "task_config_sha256": "4859b2ff2da83e993dec2e4f24e9b173eac9cdb666adc0b2dd547b38cff3f529",
        "global_step": 26000,
        "update_idx": 73489,
        "label": "直行候选：严格测试通过5/6；尚未通过课堂行走验收",
    },
}


def emit(value: dict) -> None:
    if os.environ.get("MICRODUCK_LESSON_BRIEF") == "1":
        event_path = PLAY_OUTPUTS / "launcher_events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                                     **value}, ensure_ascii=False) + "\n")
        status = value.get("status")
        if status == "dry_run":
            print("训练预览：这一步还不会开始训练。")
            print("训练起点：随机初始化模型。" if value["start"] == "random_initialization"
                  else "训练起点：课堂起步模型。")
            print(f"新增训练循环：{value['additional_vector_iterations']}；转向扣分系数：{value['heading_rate_cost']['weight']:g}")
            if value.get("constraints_file"):
                print(f"约束文件：{value['constraints_file']}（已读取并校验）")
                print("完整生效参数会随本次训练保存；experiment.yaml 的倍率不再叠加。")
            print("预览无误后，去掉命令末尾的 --dry-run 才会真正训练。")
        elif status == "training_starting":
            print("开始训练，请等待终端显示“训练完成”。")
            print(f"本次结果目录：{value['new_course_run']}")
        elif status == "completed" and value.get("new_checkpoint"):
            print("下一步：用 play-trained --latest 回放最近完成的续训。" if value.get("student") == "local_tutorial"
                  else "这是从零训练的独立结果；不会替换续训模型。")
        elif status == "starting_playback":
            print("正在播放：" + ("走偏示例" if value["case"] == "drift" else "直行候选"))
            print("正在生成回放视频，不弹出窗口。" if value["mode"] == "headless_video"
                  else "窗口打开后即可观察小鸭子；关闭仿真窗口后回到终端。")
            print("播放已有模型不会继续训练。")
        elif status == "playback_returned":
            print("回放已结束。")
            if value.get("video"):
                print(f"视频文件：{value['video']}")
        elif status == "local_files_and_dependencies_verified":
            print("环境检查通过，可以运行小鸭子。")
            print("使用的 Python：" + value["python"])
            print("这一步只检查文件和依赖，不会打开仿真窗口。")
        elif status == "failed":
            print("操作未完成：" + value.get("error", "检查未通过，请查看详细记录。"))
        else:
            print("操作状态：" + str(status or "记录已保存"))
        print(f"详细运行记录：{event_path}", flush=True)
        return
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def bound_case(name: str) -> tuple[dict, Path, Path, dict, dict]:
    """Bind policy, original task metadata and portable course settings by hash."""
    item = CASES[name]
    source = SOURCE_RUNS / item["course_run"]
    run_dir = source / "motrix_runs/microduck-course-flat/motrix/torch/fastsac" / item["upstream_run"]
    policy = run_dir / item["checkpoint"]
    files = {
        "course_config": (source / "course_config.json", item["course_config_sha256"]),
        "metadata": (run_dir / "metadata.json", item["metadata_sha256"]),
        "task_config": (run_dir / "task_config.yaml", item["task_config_sha256"]),
        "checkpoint": (policy, item["sha256"]),
    }
    hashes = {}
    for key, (path, expected) in files.items():
        actual = sha256(path)
        if actual != expected:
            raise CourseError(f"{name}源文件SHA256不匹配，停止使用：{path}")
        hashes[key] = {"path": str(path), "sha256": actual}
    validate_policy_files(run_dir, policy, item["sha256"])
    saved = json.loads((source / "course_config.json").read_text(encoding="utf-8"))
    if saved["course"]["upstream_commit"] != "1ddb9743c220c3a1e234ea0237d5ce9cc56d0499":
        raise CourseError("源课程配置的固定上游提交不匹配。")
    return saved, run_dir, policy, hashes, read_yaml(run_dir / "task_config.yaml")


def portable_settings(saved: dict) -> dict:
    """Relocate working paths in memory; keep archived source bytes untouched."""
    settings = copy.deepcopy(saved)
    settings["course"]["upstream_path"] = "upstream/MotrixLab"
    settings["paths"]["outputs"] = "outputs/training_outputs"
    return settings


def state_for(name: str, policy: Path) -> dict:
    from microduck_course.runtime import checkpoint_state
    state = checkpoint_state(policy, require_optimizers=True)
    if (state["global_step"], state["update_idx"]) != (CASES[name]["global_step"], CASES[name]["update_idx"]):
        raise CourseError(f"{name}真实检查点计数与固定身份不一致。")
    return state


def check(*, brief: bool = False, allow_cpu: bool = False) -> int:
    from microduck_course.runtime import RUNTIME_MODULES, add_upstream_sources
    results = {}
    settings = None
    failed = False
    for name in CASES:
        try:
            saved, run_dir, policy, hashes, _ = bound_case(name)
            state = state_for(name, policy)
            results[name] = {"status": "hash_and_training_state_verified", "label": CASES[name]["label"],
                             "global_step": state["global_step"], "update_idx": state["update_idx"],
                             "complete_optimizer_state": True, "files": hashes, "run_dir": str(run_dir)}
            settings = saved
        except Exception as exc:
            failed = True
            results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    dependencies = {}
    module_origins = {"microduck_course": str(Path(sys.modules["microduck_course"].__file__).resolve())}
    if settings:
        add_upstream_sources(ROOT, settings)
        for name in RUNTIME_MODULES:
            try:
                found = importlib.util.find_spec(name)
                dependencies[name] = bool(found)
                module_origins[name] = found.origin if found else None
            except (ImportError, ValueError):
                dependencies[name] = False
        failed = failed or not all(dependencies.values())
    import torch
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    try:
        versions["motrixsim"] = importlib.metadata.version("motrixsim-core")
    except importlib.metadata.PackageNotFoundError:
        versions["motrixsim"] = "distribution version unavailable"
    if versions["cuda_available"]:
        versions["gpu"] = torch.cuda.get_device_name(0)
    elif not allow_cpu:
        failed = True  # The bound training settings explicitly use CUDA.
    report = {"status": "failed" if failed else "local_files_and_dependencies_verified",
          "root": str(ROOT), "python": sys.executable, "prepared_python": str(PYTHON),
          "versions": versions, "dependencies_discoverable": dependencies, "cases": results,
          "module_origins": module_origins, "source_runs": str(SOURCE_RUNS),
          "training_outputs": str(TRAIN_OUTPUTS),
          "cuda_required": not allow_cpu,
          "simulation_executed": False, "training_executed": False, "window_verified": False,
          "walking_verified": False, "scope": "本检查不创建仿真环境，不替代窗口与行走实测"}
    if brief:
        report_path = PLAY_OUTPUTS / "environment_check.json"
        write_json(report_path, report)
        if failed:
            print("检查未通过，请先处理下列问题。", flush=True)
        elif versions["cuda_available"]:
            print("环境检查通过，可以运行小鸭子。", flush=True)
        else:
            print("CPU 环境和权重检查通过。已固定的课堂续训配置仍需要 CUDA。", flush=True)
        print(f"使用的 Python：{sys.executable}")
        print(f"训练显卡：{versions.get('gpu', 'CUDA 不可用')}")
        for name, result in results.items():
            label = "走偏示例" if name == "drift" else "直行候选"
            print(f"{label}：" + ("文件和模型已核对" if result["status"] != "failed" else result["error"]))
        for name, found in dependencies.items():
            if not found:
                print(f"缺少依赖：{name}")
        print("这一步只检查文件和依赖，不会打开仿真窗口。")
        print(f"详细检查报告：{report_path}", flush=True)
    else:
        emit(report)
    return 2 if failed else 0


def completed_training(path: Path) -> tuple[dict, dict, Path, Path, datetime, str]:
    """Read only a real, completed tutorial continuation and validate its policy."""
    if path.resolve().parent != TRAINING_GROUP.resolve():
        raise CourseError("只接受本教程 local_tutorial 目录内的续训记录。")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("student") != "local_tutorial"
            or manifest.get("run_id") != path.name):
        raise CourseError(f"该记录不是已完成的本教程续训：{path.name}")
    config_path = path / "course_config.json"
    expected_config = manifest.get("config_files", {}).get("course_config.json")
    if not expected_config or sha256(config_path) != expected_config:
        raise CourseError(f"该次续训的配置文件未登记或已变化：{path.name}")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    if settings["course"]["upstream_commit"] != "1ddb9743c220c3a1e234ea0237d5ce9cc56d0499":
        raise CourseError("该次续训没有使用本教程固定版 MotrixLab。")
    run_dir, policy, _ = course_run_policy(path, manifest)
    try:
        run_dir.resolve().relative_to(path.resolve())
    except ValueError as exc:
        raise CourseError("训练模型目录必须位于该次训练记录内。") from exc
    from microduck_course.runtime import checkpoint_state
    state = checkpoint_state(policy, require_optimizers=True)
    if (state["global_step"] != manifest.get("final_global_step")
            or state["update_idx"] != manifest.get("final_update_idx")
            or manifest.get("new_critic_updates", 0) <= 0
            or manifest.get("new_actor_update_gates", 0) <= 0):
        raise CourseError(f"该次续训的实际模型与完成计数不一致：{path.name}")
    if manifest.get("completed_at_utc"):
        finished = datetime.fromisoformat(manifest["completed_at_utc"])
        if finished.tzinfo is None:
            raise CourseError("训练完成时间缺少时区，不能确定最新结果。")
        time_source = "训练完成时间"
    else:
        # Older successful runs did not have an explicit completion timestamp.
        # Do not rewrite them or confuse directory creation time with completion.
        finished = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc)
        time_source = "旧记录的完成清单修改时间"
    return manifest, portable_settings(settings), run_dir, policy, finished, time_source


def select_completed_training(run_id: str | None):
    if run_id:
        if not re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{8}", run_id):
            raise CourseError("--run 需要填写训练完成时显示的完整编号。")
        path = TRAINING_GROUP / run_id
        return path, completed_training(path)
    valid = []
    rejected = []
    for manifest_path in TRAINING_GROUP.glob("*/manifest.json"):
        try:
            result = completed_training(manifest_path.parent)
            valid.append((manifest_path.parent, result))
        except (CourseError, OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append(f"{manifest_path.parent.name}：{exc}")
    if not valid:
        detail = "\n" + "\n".join(rejected) if rejected else ""
        raise CourseError("没有找到已完成且模型有效的教程续训。先完成 train，再运行此命令。" + detail)
    selected = max(valid, key=lambda item: (item[1][4], item[0].name))
    return selected


def play_trained(args: argparse.Namespace) -> int:
    from microduck_course.cli import perform_playback
    path, (manifest, saved, run_dir, policy, finished, time_source) = select_completed_training(args.run)
    print("正在回放：本次训练结果（最近完成的一次）。" if args.latest else "正在回放：指定的训练结果。", flush=True)
    print(f"训练编号：{manifest['run_id']}")
    print(f"选择依据：{time_source}，{finished.astimezone().isoformat(timespec='seconds')}")
    print(f"这次新增采样循环：{manifest['final_global_step'] - manifest['initial_global_step']}")
    print(f"模型文件：{policy}")
    print("这是已保存模型的回放；不会继续学习。", flush=True)
    result = perform_playback(ROOT, saved, path, run_dir, policy, record=args.record,
                              seconds=args.seconds, role="local_tutorial_completed_training")
    emit({"selected_training_run": manifest["run_id"], **result})
    return 0


def play(args: argparse.Namespace) -> int:
    from microduck_course.cli import perform_playback
    saved, run_dir, policy, _, _ = bound_case(args.case)
    state_for(args.case, policy)
    emit({"status": "starting_playback", "case": args.case, "label": CASES[args.case]["label"],
          "checkpoint": str(policy), "checkpoint_sha256": CASES[args.case]["sha256"],
          "mode": "headless_video" if args.record else "native_window_close_window_to_finish",
          "walking_verified": False})
    result = perform_playback(ROOT, portable_settings(saved), PLAY_OUTPUTS / args.case, run_dir, policy,
                              record=args.record, seconds=args.seconds, role=f"local_tutorial_{args.case}")
    emit(result)
    return 0


def train(args: argparse.Namespace) -> int:
    from microduck_course.runtime import actor_update_gates, require_learning_progress, train_real
    saved, run_dir, policy, source_hashes, source_task = bound_case("straight")
    state = state_for("straight", policy)
    settings = portable_settings(saved)
    settings["runtime"]["additional_iterations"] = args.steps
    settings["teacher_task_profile"] = PROFILES[args.yaw_weight]
    effective_constraints = None
    if args.constraints:
        from microduck_course.constraints import attach_constraints, apply_constraints, constraint_summary
        from microduck_course.runtime import add_upstream_sources
        from microduck_course.task import apply_teacher_task_profile
        settings = attach_constraints(ROOT, settings, args.constraints)
        add_upstream_sources(ROOT, settings)
        from motrix_envs.locomotion.humanoid.microduck import make_microduck_walk_flat_cfg
        task = apply_constraints(apply_teacher_task_profile(make_microduck_walk_flat_cfg(), settings["teacher_task_profile"]),
                                 settings["walking_constraints"])
        effective_constraints = constraint_summary(task)
    # Preserve the full audited source algorithm even for random initialization.
    # train_real adjusts the absolute trainer target from the real initial step.
    settings["fastsac"] = copy.deepcopy(source_task["algo"])
    visual = copy.deepcopy(saved["training_visualization"])
    visual["enabled"] = args.render
    settings["training_visualization"] = training_visualization_options(visual)
    algorithm = resolved_algorithm(ROOT, settings, None if args.from_scratch else run_dir)
    if algorithm["agent"] != source_task["algo"]["agent"]:
        raise CourseError("续训算法agent与已审计源配置不一致，拒绝运行。")
    agent = algorithm["agent"]
    initial_step = 0 if args.from_scratch else state["global_step"]
    initial_update = 0 if args.from_scratch else state["update_idx"]
    expected_updates = max(0, args.steps - agent["learning_starts"]) * agent["num_updates"]
    expected = require_learning_progress(initial_update, initial_update + expected_updates, agent["policy_frequency"])
    target = initial_step + args.steps
    plan = {
        "status": "dry_run" if args.dry_run else "training_planned",
        "start": "random_initialization" if args.from_scratch else "fixed_26k_complete_training_state",
        "source_files": source_hashes, "source_agent_matches": True,
        "additional_vector_iterations": args.steps, "num_envs": settings["runtime"]["num_envs"],
        "new_environment_transitions": args.steps * settings["runtime"]["num_envs"],
        "initial_global_step": initial_step, "target_global_step": target,
        "initial_update_idx": initial_update,
        "expected_final_update_idx": initial_update + expected_updates,
        "expected_new_critic_updates": expected_updates,
        "expected_new_actor_update_gates": actor_update_gates(initial_update, initial_update + expected_updates, agent["policy_frequency"]),
        "learning_starts_per_run": agent["learning_starts"],
        "heading_rate_cost": {"weight": effective_constraints["rewards"]["penalty_heading_rate"]["weight"] if effective_constraints else args.yaw_weight,
                              "sigma": effective_constraints["rewards"]["penalty_heading_rate"]["sigma"] if effective_constraints else 0.015,
                              "profile": settings["teacher_task_profile"]},
        "tracking_ang_vel_sigma": effective_constraints["rewards"]["tracking_ang_vel"]["sigma"] if effective_constraints else 0.15,
        "constraints_file": args.constraints,
        "effective_task_constraints": effective_constraints,
        "training_visualization": settings["training_visualization"],
        "output_base": str(TRAIN_OUTPUTS), "walking_verified": False,
        "limitations": ["steps是新增向量采样外循环，非梯度更新数或视频帧数",
                        "短训只验证真实更新流程，不保证行走或改善；从零训练也不保证学会行走",
                        "render是现有真实训练窗口；训练探索动作与固定策略评价不同"],
    }
    if args.dry_run:
        emit({**plan, "training_executed": False, "checkpoint_created": False, "output_created": False})
        return 0
    source_run = None if args.from_scratch else run_dir
    source_policy = None if args.from_scratch else policy
    path, manifest = new_run(ROOT, settings, "local_tutorial_scratch" if args.from_scratch else "local_tutorial")
    manifest.update(start={"kind": "random_initialization", "seed": settings["runtime"]["seed"]} if args.from_scratch
                    else policy_provenance(run_dir, policy, "local_tutorial_fixed_26k_continuation"),
                    tutorial_plan=plan, source_files=source_hashes)
    try:
        write_json(path / "manifest.json", manifest)
        write_json(path / "source_course_config.json", saved)
        write_json(path / "algorithm_audit.json", {"source_agent_matches": True, "source_agent": source_task["algo"]["agent"],
                                                   "resolved_agent": algorithm["agent"], "expected_learning": expected})
        manifest["config_files"] = {name: sha256(path / name) for name in ("source_course_config.json", "course_config.json", "algorithm_audit.json")}
        write_json(path / "manifest.json", manifest)
        emit({"status": "training_starting", "new_course_run": str(path), "plan": plan})
        manifest.update(train_real(ROOT, settings, path, source_run, source_policy))
        # Verify the original policy/configuration again after learning.
        bound_case("straight")
        manifest.update(status="completed", source_unchanged=True,
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(path / "manifest.json", manifest)
        raise
    write_json(path / "manifest.json", manifest)
    checkpoint = path / manifest["upstream_run_dir"] / manifest["checkpoint"]
    if not args.from_scratch:
        # Publish the convenience pointer only after successful real training.
        write_json(LAST_TRAINING, {"run_id": manifest["run_id"], "student": "local_tutorial",
                                  "completed_at_utc": manifest["completed_at_utc"],
                                  "checkpoint_sha256": manifest["checkpoint_sha256"]})
    print("\n训练完成，已保存新模型。", flush=True)
    print(f"训练编号：{manifest['run_id']}")
    print(f"本次新增采样循环：{manifest['final_global_step'] - initial_step}")
    print(f"模型文件：{checkpoint}", flush=True)
    emit({"new_course_run": str(path), "new_checkpoint": str(checkpoint), **manifest,
          "play_command": ([str(ROOT / "tools/lesson.cmd"), "play-trained", "--run", manifest["run_id"]]
                           if not args.from_scratch else
                           [str(PYTHON), "-m", "microduck_course", "play", "--classroom", "configs/teacher_heading_continue28.yaml",
                            "--student", manifest["student"], "--run", manifest["run_id"]])})
    return 0


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def yaw_weight(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result not in PROFILES:
        raise argparse.ArgumentTypeError("现有真实profile仅支持-15、-20、-25；默认-20")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="本机真实MotrixSim教程：查看走偏示例和直行候选，从课堂起步模型继续训练。")
    commands = p.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="只读核对固定权重、完整训练状态、依赖与CUDA；不创建仿真")
    checking.add_argument("--brief", action="store_true", help="显示简洁中文结论，并保存完整检查报告")
    checking.add_argument("--allow-cpu", action="store_true", help="无 CUDA 时仍检查依赖和权重，供 CPU 环境验证")
    playback = commands.add_parser("play", help="默认打开真实交互窗口，关闭窗口结束")
    playback.add_argument("--case", required=True, choices=tuple(CASES))
    playback.add_argument("--record", action="store_true", help="改为真实离屏录像；不会打开交互窗口")
    playback.add_argument("--seconds", type=positive_int, default=20, help="仅控制录像秒数，默认20")
    trained = commands.add_parser("play-trained", help="回放本教程已完成的续训模型，不再手工拼配置和路径")
    selection = trained.add_mutually_exclusive_group(required=True)
    selection.add_argument("--latest", action="store_true", help="按完成时间选择最近完成且模型有效的本教程续训")
    selection.add_argument("--run", help="指定该教程续训的完整编号")
    trained.add_argument("--record", action="store_true", help="录制离屏视频，不打开交互窗口")
    trained.add_argument("--seconds", type=positive_int, default=20, help="录像秒数，默认20")
    training = commands.add_parser("train", help="默认从课堂起步模型的完整状态续训，另建记录，不覆盖旧权重")
    training.add_argument("--steps", required=True, type=positive_int, help="新增向量采样外循环；每轮512环境，需跨过64轮learning_starts")
    constraint_source = training.add_mutually_exclusive_group()
    constraint_source.add_argument("--yaw-weight", type=yaw_weight, default=-20.0, help="转向扣分系数，仅-15/-20/-25；与--constraints二选一")
    constraint_source.add_argument("--constraints", help="加载完整行走约束 YAML，覆盖任务预设；不叠加 experiment.yaml 倍率")
    training.add_argument("--from-scratch", action="store_true", help="保留相同算法设置但随机初始化；不保证学会行走")
    training.add_argument("--render", action="store_true", help="启用现有TrainingView的真实原生训练窗口和env0探索录像")
    training.add_argument("--dry-run", action="store_true", help="校验真实来源，显示预计计数/目标；不创建run，不启动训练/窗口")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "check":
            return check(brief=args.brief, allow_cpu=args.allow_cpu)
        if args.command == "play-trained":
            return play_trained(args)
        return play(args) if args.command == "play" else train(args)
    except (CourseError, OSError, ImportError, ValueError, KeyError) as exc:
        emit({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
