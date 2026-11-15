"""Classroom commands with explicit plans, failures and real artifact outputs."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from microduck_course.config import CourseError, evaluation_settings, load_settings, project_root, read_yaml, resolved_algorithm, resolve_path, validate_student
from microduck_course.preflight import inspect_readiness
from microduck_course.storage import course_run_policy, new_run, open_course_run, policy_provenance, resolve_registered_policy, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="MicroDuck 行走课程：官方 MotrixLab FastSAC + MotrixSim。")
    commands = result.add_subparsers(dest="command", required=True)
    descriptions = {
        "config": "显示真实合并配置与静态策略契约",
        "check": "只读课前检查；--runtime 在已构建环境中实测 reset/维度",
        "baseline": "教师参考策略示范；受控实验基线需用倍率 1.0 独立续训",
        "train": "从教师 starter 真正续训；--from-scratch 供教师准备",
        "play": "用该次运行保存的配置与策略在 MotrixSim 回放",
        "export": "调用官方 ONNX 导出和数值 parity 校验",
        "collect": "将真实完成的运行和结果打包提交",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--root", help="工程根目录；通常无需指定")
        command.add_argument("--classroom", default="configs/classroom.yaml")
        command.add_argument("--config", default="configs/experiment.yaml")
        command.add_argument("--constraints", help="统一行走约束 YAML；提供后以其中参数为准，不再叠加 experiment 倍率")
        command.add_argument("--student", default="student01", help="匿名学员代号，例如 s01")
        if name in ("baseline", "train", "play", "export"):
            command.add_argument("--dry-run", action="store_true", help="只输出计划，不运行模拟、不训练、不生成权重")
        if name in ("play", "export", "collect"):
            command.add_argument("--run", required=True, help="当前学员已有训练输出的 run_id")
        if name == "train":
            start = command.add_mutually_exclusive_group()
            start.add_argument("--from-scratch", action="store_true", help="教师从零训练，课堂内不保证学会行走")
            start.add_argument("--teacher-resume-run", help="教师继续本人已完成且哈希通过的训练run；不将它登记为课堂starter")
            command.add_argument("--teacher-source-student", help="教师续训源代号；默认与--student相同")
        if name == "check":
            command.add_argument("--runtime", action="store_true", help="实际构建 MotrixSim 环境并 reset；不训练、不验收步态")
        if name in ("baseline", "play"):
            command.add_argument("--record", action="store_true", help="录制真实 MotrixSim 视频，替代交互窗口")
            command.add_argument("--seconds", type=int, default=20, help="录制时长，默认 20 秒")
    return result


def emit(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def perform_playback(root: Path, settings: dict, parent: Path, run_dir: Path, checkpoint: Path, *, record: bool, seconds: int, role: str) -> dict:
    from microduck_course.runtime import play_real
    evaluation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = parent / "evaluations" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    playback = evaluation_settings(settings)
    snapshot = destination / "evaluation_settings.json"
    write_json(snapshot, playback)
    report = {"status": "running", "source": policy_provenance(run_dir, checkpoint, role), "evaluation": playback["evaluation"], "walking_verified": False, "velocity_metrics_computed": False}
    report_path = destination / "evaluation_report.json"
    write_json(report_path, report)
    video = destination / "walking.mp4" if record else None
    try:
        report.update(play_real(root, playback, snapshot, run_dir, checkpoint, video=video, seconds=seconds))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(report_path, report)
        raise
    report["video"] = str(video) if video else None
    write_json(report_path, report)
    return {"evaluation_dir": str(destination), **report}


def run(args: argparse.Namespace) -> int:
    root = project_root(args.root)
    validate_student(args.student)
    settings = load_settings(root, args.classroom, args.config)
    if args.constraints:
        from microduck_course.constraints import attach_constraints
        settings = attach_constraints(root, settings, args.constraints)
    if args.command == "config":
        contract = read_yaml(root / "configs/policy_contract.yaml")
        emit({"settings": settings, "derived": {
            "status": "static_expectation_not_runtime_measurement",
            "expected_action_rate_weight": (settings["walking_constraints"]["task"]["rewards"]["penalty_action_rate"]["weight"]
                if "walking_constraints" in settings else contract["reward_experiment"]["statically_derived_base_weight"] * settings["experiment"]["action_rate_multiplier"]),
            "expected_static_spaces": {"policy": contract["policy_observation"]["expected_dimension"], "value": contract["value_observation"]["expected_dimension"], "action": contract["action"]["expected_dimension"]},
        }, "from_scratch_resolved_algorithm": resolved_algorithm(root, settings), "algorithm_merge_order": ["official_base", "official_microduck_task", "starter_actual_algo_if_present", "classroom"], "policy_contract": contract})
        return 0
    if args.command == "check":
        report = inspect_readiness(root, settings)
        if args.runtime:
            from microduck_course.runtime import runtime_probe
            try:
                report["runtime"] = runtime_probe(root, settings)
            except Exception as exc:
                report["runtime"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                report["status"] = "blocked"
        emit(report)
        return 2 if report["status"] == "blocked" else 0
    if args.command == "train":
        if args.teacher_source_student and not args.teacher_resume_run:
            raise CourseError("--teacher-source-student 需与 --teacher-resume-run 一起使用。")
        if args.dry_run:
            emit({"status": "dry_run", "training_executed": False, "checkpoint_created": False, "task": settings["course"]["task"], "simulator": "motrixsim", "algorithm": "official_motrix.fastsac", "student": args.student, "start": "random_initialization" if args.from_scratch else "teacher_continuation" if args.teacher_resume_run else "teacher_starter_required", "teacher_source_run": args.teacher_resume_run, "action_rate_multiplier": settings["experiment"]["action_rate_multiplier"], "additional_iterations": settings["runtime"]["additional_iterations"], "num_envs": settings["runtime"]["num_envs"], "preflight": inspect_readiness(root, settings)})
            return 0
        starter_run = starter_checkpoint = None
        if args.teacher_resume_run:
            source_student = validate_student(args.teacher_source_student or args.student)
            source_path, source_manifest = open_course_run(root, settings, source_student, args.teacher_resume_run)
            starter_run, starter_checkpoint, _ = course_run_policy(source_path, source_manifest)
        elif not args.from_scratch:
            starter_run, starter_checkpoint, _ = resolve_registered_policy(root, settings, "starter")
        path, manifest = new_run(root, settings, args.student)
        try:
            manifest["start"] = {"kind": "random_initialization", "seed": settings["runtime"]["seed"]} if args.from_scratch else policy_provenance(starter_run, starter_checkpoint, "teacher_continuation_not_classroom_starter" if args.teacher_resume_run else "starter")
            write_json(path / "manifest.json", manifest)
            from microduck_course.runtime import train_real
            manifest.update(train_real(root, settings, path, starter_run, starter_checkpoint))
            manifest["status"] = "completed"
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(path / "manifest.json", manifest)
            raise
        write_json(path / "manifest.json", manifest)
        emit({"run_dir": str(path), **manifest})
        return 0
    if args.command == "baseline":
        if args.dry_run:
            emit({"status": "dry_run", "simulation_executed": False, "policy": "reference", "purpose": "teacher_demonstration_not_controlled_experiment_baseline", "simulator": "motrixsim", "evaluation": settings["evaluation"], "prerequisites": inspect_readiness(root, settings)["policies"]["reference"]})
            return 0
        run_dir, checkpoint, _ = resolve_registered_policy(root, settings, "reference")
        parent = resolve_path(root, settings["paths"]["outputs"]) / args.student / "reference"
        emit(perform_playback(root, settings, parent, run_dir, checkpoint, record=args.record, seconds=args.seconds, role="reference"))
        return 0
    path, manifest = open_course_run(root, settings, args.student, args.run)
    saved = json.loads((path / "course_config.json").read_text(encoding="utf-8"))
    run_dir, checkpoint, _ = course_run_policy(path, manifest)
    if getattr(args, "dry_run", False):
        emit({"status": "dry_run", "operation": args.command, "simulation_executed": False, "checkpoint": str(checkpoint), "simulator": "motrixsim", "saved_action_rate_multiplier": saved["experiment"]["action_rate_multiplier"], "evaluation": saved["evaluation"]})
        return 0
    if args.command == "play":
        emit(perform_playback(root, saved, path, run_dir, checkpoint, record=args.record, seconds=args.seconds, role="trained"))
    elif args.command == "export":
        from microduck_course.runtime import export_real
        output = path / "export/policy.onnx"
        if output.exists():
            raise CourseError(f"导出已存在，请先另存以保留本次结果: {output}")
        emit(export_real(root, saved, path / "course_config.json", run_dir, checkpoint, manifest["checkpoint_sha256"], output))
    elif args.command == "collect":
        output = resolve_path(root, settings["paths"]["submissions"]) / f"{args.student}-{args.run}.zip"
        output.parent.mkdir(parents=True, exist_ok=True)
        # Always package only this student's selected run, never account/config secrets.
        if output.exists():
            raise CourseError(f"提交包已存在: {output}")
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(path.rglob("*")):
                if file.is_file() and not file.is_symlink() and file.suffix not in (".tmp", ".env"):
                    archive.write(file, file.relative_to(path).as_posix())
        emit({"status": "collected", "submission": str(output), "training_status": manifest["status"], "walking_verified": manifest.get("walking_verified", False)})
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "seconds", 20) <= 0:
        print("ERROR: --seconds 必须大于 0。", file=sys.stderr)
        return 2
    try:
        return run(args)
    except (CourseError, OSError, ImportError, ValueError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
