"""Record actual FastSAC checkpoints through the course's real MotrixSim player.

No training, checkpoint synthesis, or alternate policy implementation lives here.
Heavy dependencies are imported only after argument parsing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

FPS = 30  # The existing play_real API fixes this value.


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def parse_steps(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--steps 格式为 100,200,400，使用 checkpoint 内的 global_step") from exc
    if not values or any(number < 0 for number in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--steps 必须是不重复的非负整数")
    return sorted(values)


def select_stages(inventory: list[dict], steps: list[int] | None) -> list[dict]:
    ordered = sorted(inventory, key=lambda item: (item["global_step"], item["update_idx"], item["checkpoint"]))
    if steps is not None:
        selected = []
        for step in steps:
            matches = [item for item in ordered if item["global_step"] == step]
            if not matches:
                available = sorted({item["global_step"] for item in ordered})
                raise ValueError(f"没有 global_step={step} 的真实 checkpoint；现有值: {available}")
            if len(matches) != 1:
                raise ValueError(f"global_step={step} 对应不同内容的 checkpoint，无法唯一选择: {[item['checkpoint'] for item in matches]}")
            selected.append(matches[0])
        return selected
    count = min(5, len(ordered))
    if count <= 1:
        return ordered
    # Quantiles of actual files, never invented step numbers.
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def add_course_source(root: Path) -> None:
    source = root / "src"
    if not (source / "microduck_course/runtime.py").is_file():
        raise ValueError(f"找不到课程工程: {root}")
    sys.path.insert(0, str(source))


def discover_checkpoints(run_dir: Path) -> tuple[list[dict], list[dict]]:
    from microduck_course.runtime import checkpoint_state
    from microduck_course.storage import sha256, validate_policy_files

    directory = run_dir / "checkpoints"
    candidates: dict[Path, list[str]] = {}

    def add(path: Path, label: str) -> None:
        resolved = path.resolve()
        resolved.relative_to(run_dir)
        if resolved.suffix != ".pt":
            raise ValueError(f"仅支持当前 FastSAC 的 .pt checkpoint: {resolved}")
        candidates.setdefault(resolved, []).append(label)

    for path in sorted(directory.glob("model_*.pt")):
        if re.fullmatch(r"model_\d+\.pt", path.name):
            add(path, "periodic")
    if (directory / "latest.pt").exists():
        add(directory / "latest.pt", "standard_final")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        artifacts = read_json(manifest_path).get("artifacts", {})
        for role in ("latest_training_state", "best_policy"):
            if role in artifacts:
                add(directory / artifacts[role]["path"], role)
    inventory, errors = [], []
    by_hash: dict[str, dict] = {}
    for path, roles in candidates.items():
        relative = path.relative_to(run_dir).as_posix()
        try:
            digest = sha256(path)
            validate_policy_files(run_dir, path, digest)
            state = checkpoint_state(path)
            step, update = state["global_step"], state["update_idx"]
            del state
            if sha256(path) != digest:
                raise ValueError("checkpoint 在检查期间发生变化，请等待训练/保存结束")
            if digest in by_hash:
                by_hash[digest]["aliases"].append({"checkpoint": relative, "roles": roles})
                continue
            item = {
                "checkpoint": relative, "checkpoint_sha256": digest,
                "checkpoint_bytes": path.stat().st_size, "global_step": step,
                "update_idx": update, "roles": roles, "aliases": [],
            }
            inventory.append(item)
            by_hash[digest] = item
        except Exception as exc:
            errors.append({"checkpoint": relative, "status": "failed_inspection", "error": f"{type(exc).__name__}: {exc}"})
    return inventory, errors


def inspect_video(path: Path, expected_frames: int) -> dict:
    """Decode every video frame; a writer return/file header alone is insufficient."""
    result = {"path": str(path), "status": "failed", "expected_frame_count": expected_frames,
              "actual_frame_count": None, "actual_fps": None, "file_size_bytes": 0}
    try:
        import imageio_ffmpeg

        result["file_size_bytes"] = path.stat().st_size
        if not result["file_size_bytes"]:
            raise ValueError("视频文件为空")
        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            metadata = next(reader)
        finally:
            reader.close()
        result["actual_fps"] = float(metadata["fps"])
        result["size"] = list(metadata["size"])
        result["container_duration_seconds"] = metadata.get("duration")
        # -vsync 0 prevents FFmpeg from manufacturing/dropping frames during this check.
        process = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-xerror",
            "-err_detect", "explode", "-i", str(path), "-map", "0:v:0", "-an",
            "-vsync", "0", "-f", "null", "-", "-progress", "pipe:1", "-nostats",
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        frames = re.findall(r"^frame=(\d+)\s*$", process.stdout, flags=re.MULTILINE)
        result["actual_frame_count"] = int(frames[-1]) if frames else 0
        result["decoder_returncode"] = process.returncode
        result["decoder_errors"] = process.stderr[-6000:]
        result["decoded_to_end"] = process.returncode == 0 and "progress=end" in process.stdout
        if not result["decoded_to_end"] or process.stderr.strip():
            raise ValueError("FFmpeg 未能无错误地完整解码视频")
        if result["actual_frame_count"] != expected_frames:
            raise ValueError(f"实际帧数 {result['actual_frame_count']}，预期 {expected_frames}；拒绝将部分视频标记成功")
        if not math.isclose(result["actual_fps"], FPS, abs_tol=0.01):
            raise ValueError(f"实际 FPS={result['actual_fps']}，当前播放器要求 {FPS}")
        if result["size"] != [960, 540]:
            raise ValueError(f"实际视频尺寸异常: {result['size']}")
        result["encoded_video_duration_seconds"] = result["actual_frame_count"] / result["actual_fps"]
        result["status"] = "verified_complete_video"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def create_contact_sheet(video: Path, inspection: dict, output: Path) -> dict:
    """Select five actual decoded frames, with their frame/time labels."""
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from microduck_course.storage import sha256

    frame_count = inspection["actual_frame_count"]
    indices = sorted({round(fraction * (frame_count - 1)) for fraction in (0, 0.25, 0.5, 0.75, 1)})
    frames = {}
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        for index, raw in enumerate(reader):
            if index in indices:
                frames[index] = Image.frombytes("RGB", tuple(metadata["size"]), raw)
            if index >= indices[-1]:
                break
    finally:
        reader.close()
    if len(frames) != len(indices):
        raise ValueError(f"Contact sheet 缺少真实帧: wanted={indices}, decoded={sorted(frames)}")
    width, height, label_height = 400, 225, 30
    sheet = Image.new("RGB", (width * 3, (height + label_height) * 2), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for tile, index in enumerate(indices):
        x, y = (tile % 3) * width, (tile // 3) * (height + label_height)
        sheet.paste(frames[index].resize((width, height), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 8, y + height + 7), f"frame {index} | {index / inspection['actual_fps']:.2f} s", fill="black")
    sheet.save(output)
    with Image.open(output) as saved:
        saved.verify()
    return {"path": str(output), "sha256": sha256(output), "file_size_bytes": output.stat().st_size,
            "source": "actual_decoded_video_frames", "frame_indices_zero_based": indices,
            "frame_times_seconds": [index / inspection["actual_fps"] for index in indices],
            "walking_verified": False}


def validate_source(request: dict, checkpoint: Path) -> None:
    from microduck_course.runtime import checkpoint_state
    from microduck_course.storage import sha256, validate_policy_files

    run_dir = Path(request["run_dir"])
    if sha256(Path(request["settings_path"])) != request["settings_sha256"]:
        raise ValueError("评估设置快照在登记后发生变化")
    validate_policy_files(run_dir, checkpoint, request["stage"]["checkpoint_sha256"])
    for name, expected in request["source_file_hashes"].items():
        if sha256(run_dir / name) != expected:
            raise ValueError(f"源运行 {name} 在检查后改变；拒绝混用不同运行配置")
    state = checkpoint_state(checkpoint)
    for name in ("global_step", "update_idx"):
        if state[name] != request["stage"][name]:
            raise ValueError(f"checkpoint 内容 {name} 与已登记阶段不符")


def record_worker(request_path: Path) -> int:
    request = read_json(request_path)
    destination = request_path.parent
    report_path = destination / "report.json"
    root = Path(request["root"])
    add_course_source(root)
    from microduck_course.storage import write_json

    started = time.monotonic()
    report = {**request["stage"], "status": "running", "success": False,
              "started_utc": utc_now(), "walking_verified": False,
              "velocity_metrics_computed": False, "training_executed": False,
              "settings_sha256": request["settings_sha256"],
              "requested_video_duration_seconds": request["seconds"], "requested_fps": FPS}
    write_json(report_path, report)
    try:
        from microduck_course.config import COURSE_TASK
        from microduck_course.runtime import play_real
        from microduck_course.storage import policy_provenance

        run_dir = Path(request["run_dir"])
        checkpoint = run_dir / request["stage"]["checkpoint"]
        validate_source(request, checkpoint)
        settings_path = Path(request["settings_path"])
        settings = read_json(settings_path)
        report["evaluation"] = settings["evaluation"]
        report["source"] = policy_provenance(run_dir, checkpoint, "training_stage")
        video = destination / "walking.mp4"
        playback_start = time.monotonic()
        report["playback"] = play_real(root, settings, settings_path, run_dir, checkpoint,
                                       video=video, seconds=request["seconds"])
        report["playback_wall_clock_seconds"] = time.monotonic() - playback_start
        validate_source(request, checkpoint)
        from motrix_env_core import registry

        control_dt = registry.make_env_config(COURSE_TASK).ctrl_dt
        report["simulation_control_dt_seconds"] = control_dt
        report["actual_simulation_control_steps"] = None
        report["simulation_time_note"] = (
            "播放器按控制时间累计至目标视频帧数；未暴露实际控制步计数。"
            "视频秒数不等于录制墙钟时间，也不等于训练迭代数。")
        report["video"] = inspect_video(video, request["seconds"] * FPS)
        if report["video"]["status"] != "verified_complete_video":
            raise ValueError(report["video"].get("error", "视频验证失败"))
        report["contact_sheet"] = create_contact_sheet(video, report["video"], destination / "contact_sheet.png")
        report["nominal_simulated_video_interval_seconds"] = report["video"]["encoded_video_duration_seconds"]
        validate_source(request, checkpoint)
        report.update(status="completed", success=True)
    except BaseException as exc:
        report.update(status="failed", success=False, error=f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
        video = destination / "walking.mp4"
        report["partial_video_exists"] = video.is_file()
        report["partial_video_bytes"] = video.stat().st_size if video.is_file() else 0
    finally:
        report["worker_wall_clock_seconds"] = time.monotonic() - started
        report["finished_utc"] = utc_now()
        write_json(report_path, report)
    return 0 if report["success"] else 1


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按真实 FastSAC 检查点录制 MotrixSim 阶段视频；不训练、不生成假检查点。")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="课程工程根目录")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="已完成的课程 run_id；同时指定 --student")
    source.add_argument("--run-dir", help="显式指定含 metadata.json 的上游真实训练 run_dir")
    parser.add_argument("--student", help="课程匿名学员代号，如 teacher01")
    parser.add_argument("--classroom", default="configs/classroom.yaml")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--settings", help="仅用于 --run-dir：指定该训练运行保存的 course_config.json")
    parser.add_argument("--steps", type=parse_steps, help="精确选择 checkpoint 内的 global_step，如 100,200,400")
    parser.add_argument("--seconds", type=positive_int, default=20, help="每段视频秒数；固定30 FPS，默认20秒")
    parser.add_argument("--timeout", type=positive_int, default=900, help="每段录制子进程最多墙钟秒数，默认900")
    parser.add_argument("--output", help="新建的输出目录；为防止覆盖，必须不存在")
    parser.add_argument("--list", action="store_true", help="读取真实检查点并显示选择结果，不渲染、不写输出目录")
    return parser


def resolve_source(args: argparse.Namespace, root: Path) -> tuple[Path, Path, dict, dict]:
    from microduck_course.config import load_settings, resolve_path
    from microduck_course.storage import course_run_policy, open_course_run

    settings = load_settings(root, args.classroom, args.config)
    if args.run:
        if not args.student or args.settings:
            raise ValueError("--run 必须同时使用 --student，且不能覆盖该运行已保存的 --settings")
        parent, manifest = open_course_run(root, settings, args.student, args.run)
        run_dir, _, _ = course_run_policy(parent, manifest)
        saved_path = parent / "course_config.json"
        settings = read_json(saved_path)
        source = {"kind": "completed_course_run", "student": args.student, "run": args.run,
                  "course_manifest": str(parent / "manifest.json"), "settings_source": str(saved_path)}
        return run_dir.resolve(), parent, settings, source
    run_dir = resolve_path(root, args.run_dir)
    source = {"kind": "explicit_upstream_run", "settings_source": "current_classroom_and_experiment_configs"}
    if args.settings:
        settings_path = resolve_path(root, args.settings)
        settings = read_json(settings_path)
        source["settings_source"] = str(settings_path)
    else:
        # Recover the exact course settings when run_dir sits inside a course run.
        for parent in run_dir.parents:
            saved_path, manifest_path = parent / "course_config.json", parent / "manifest.json"
            if saved_path.is_file() and manifest_path.is_file():
                manifest = read_json(manifest_path)
                if "upstream_run_dir" in manifest and resolve_path(parent, manifest["upstream_run_dir"]) == run_dir:
                    if manifest.get("status") != "completed":
                        raise ValueError("该上游 run_dir 属于尚未 completed 的课程运行")
                    course_run_policy(parent, manifest)
                    settings = read_json(saved_path)
                    source["settings_source"] = str(saved_path)
                    break
    return run_dir, run_dir, settings, source


def main(argv: list[str] | None = None) -> int:
    # Internal mode is intentionally omitted from the public parser.
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--_worker":
        return record_worker(Path(argv[1]))
    args = arguments().parse_args(argv)
    root = Path(args.root).resolve()
    add_course_source(root)
    from microduck_course.config import evaluation_settings
    from microduck_course.storage import sha256, write_json

    run_dir, parent, settings, source = resolve_source(args, root)
    source_hashes = {name: sha256(run_dir / name) for name in ("metadata.json", "task_config.yaml")}
    inventory, discovery_errors = discover_checkpoints(run_dir)
    if any(sha256(run_dir / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("源 metadata/task_config 在检查点枚举过程中发生变化")
    if not inventory:
        raise ValueError(f"没有可读的真实 FastSAC 检查点。检查错误: {discovery_errors}")
    selected = select_stages(inventory, args.steps)
    settings = evaluation_settings(settings)
    summary = {
        "schema": 1, "status": "planned", "created_utc": utc_now(), "root": str(root),
        "run_dir": str(run_dir), "source": source, "inventory": inventory,
        "discovery_errors": discovery_errors, "available_checkpoint_count": len(inventory),
        "selected_checkpoint_count": len(selected), "requested_steps": args.steps,
        "selected": selected, "evaluation": settings["evaluation"],
        "requested_fps": FPS, "requested_video_duration_seconds": args.seconds,
        "expected_frames_per_video": args.seconds * FPS,
        "source_file_hashes": source_hashes,
        "count_note": "最多选5个实际阶段；只有1–2个就只录1–2个，不生成缺失阶段。相同SHA256的别名合并。",
        "counter_semantics": {
            "global_step": "checkpoint 内的训练外循环/采样迭代计数；不是视频帧数，也不是累计梯度更新数",
            "update_idx": "checkpoint 内 FastSAC 梯度更新循环计数；actor 还受 policy_frequency 门控",
            "video_duration": "实际解码帧数 / 实际FPS，按仿真控制时间录制",
            "wall_clock": "真实录制/进程耗时，依硬件与渲染速度而定",
        }, "walking_verified": False, "velocity_metrics_computed": False,
        "training_executed": False, "sim2sim_completed": False,
    }
    if args.list:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if discovery_errors else 0
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = Path(args.output).resolve() if args.output else parent / "evaluations" / ("training-stages-" + identifier)
    # Recording artifacts must not overwrite the original source run or its config.
    for source_file in (run_dir, run_dir / "checkpoints"):
        if output == source_file:
            raise ValueError("输出必须是新的独立子目录，不能覆盖源 run_dir/checkpoints")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "evaluation_settings.json"
    write_json(snapshot, settings)
    summary.update(status="running", output_dir=str(output), settings_sha256=sha256(snapshot), stages=[])
    manifest_path = output / "manifest.json"
    write_json(manifest_path, summary)
    started = time.monotonic()
    try:
        for index, stage in enumerate(selected, start=1):
            destination = output / f"{index:02d}-step-{stage['global_step']}-update-{stage['update_idx']}"
            destination.mkdir()
            request_path = destination / "request.json"
            request = {"root": str(root), "run_dir": str(run_dir), "stage": stage,
                       "settings_path": str(snapshot), "seconds": args.seconds,
                       "settings_sha256": summary["settings_sha256"],
                       "source_file_hashes": summary["source_file_hashes"]}
            write_json(request_path, request)
            print(f"[{index}/{len(selected)}] step={stage['global_step']} update={stage['update_idx']} -> {destination}", flush=True)
            stage_started = time.monotonic()
            error, returncode = None, None
            try:
                with (destination / "render.log").open("w", encoding="utf-8") as log:
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", str(request_path)],
                                             stdout=log, stderr=subprocess.STDOUT, cwd=root,
                                             env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                                             timeout=args.timeout, check=False)
                returncode = process.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                error = f"{type(exc).__name__}: {exc}"
            report_path = destination / "report.json"
            try:
                report = read_json(report_path)
            except (OSError, ValueError) as exc:
                report = {**stage, "status": "failed", "success": False, "error": f"子进程没有有效报告: {exc}"}
            if error or returncode != 0 or report.get("status") != "completed" or report.get("success") is not True:
                report.update(status="failed", success=False)
                report["process_error"] = error or f"子进程返回码 {returncode}，报告状态 {report.get('status')}"
            report["subprocess_returncode"] = returncode
            report["subprocess_wall_clock_seconds"] = time.monotonic() - stage_started
            report["report_path"] = str(report_path)
            write_json(report_path, report)
            summary["stages"].append(report)
            write_json(manifest_path, summary)
        failures = [item for item in summary["stages"] if not item.get("success")]
        unchanged = (sha256(snapshot) == summary["settings_sha256"] and
                     all(sha256(run_dir / name) == digest for name, digest in source_hashes.items()))
        summary["settings_and_source_files_unchanged_after_recording"] = unchanged
        summary["status"] = "partial_failure" if failures or discovery_errors or not unchanged else "completed"
        summary["successful_video_count"] = sum(item.get("success") is True for item in summary["stages"])
        summary["failed_video_count"] = len(failures)
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["total_wall_clock_seconds"] = time.monotonic() - started
        summary["finished_utc"] = utc_now()
        write_json(manifest_path, summary)
    print(json.dumps({"status": summary["status"], "manifest": str(manifest_path),
                      "successful_video_count": summary["successful_video_count"],
                      "failed_video_count": summary["failed_video_count"],
                      "walking_verified": False}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError, KeyError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
