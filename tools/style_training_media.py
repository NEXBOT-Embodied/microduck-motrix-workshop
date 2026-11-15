"""Add an honest teaching frame around a verified MotrixSim recording.

No training, simulation, trimming, resizing, or invented native controls.
Requires Pillow and imageio-ffmpeg in the Python used to invoke this tool.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageChops, ImageDraw, ImageFont


CANVAS = (1280, 720)
VIEW = (20, 88, 960, 540)
DEFAULT_FONT = Path("C:/Windows/Fonts/msyh.ttc")
COLORS = {"bg": "#20242a", "panel": "#272c33", "line": "#3b424c",
          "text": "#e6ebf1", "muted": "#a4afbd", "accent": "#589ee6"}
OBSERVATION_POINTS = ("平衡 · 是否站稳", "前进 · 是否跟随命令", "平滑 · 动作是否连贯")
OBSERVATION_PROMPT = "观察：是否站稳？是否跟随前进命令？动作是否平滑？"
WALKING_PROMPT = "行走效果待观察"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} 缺少带时区的真实时间")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} 必须包含时区")
    return parsed.astimezone(timezone.utc)


def integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
    return value


def digest_value(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name} 必须是 SHA256")
    return value.lower()


def load_report(stage: Path, supplied_hash: str | None) -> tuple[dict, dict]:
    report_path = stage / "report.json"
    video_path = stage / "walking.mp4"
    report_hash = sha256(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("success") is not True or report.get("status") != "completed":
        raise ValueError("只接受 success=true 且 status=completed 的真实阶段报告")
    video = report["video"]
    if video.get("status") != "verified_complete_video" or video.get("decoded_to_end") is not True:
        raise ValueError("阶段报告尚未验证原始视频完整解码")
    if Path(video["path"]).resolve() != video_path.resolve():
        raise ValueError("report.video.path 与 stage/walking.mp4 不一致")
    frames = integer(video["actual_frame_count"], "actual_frame_count", 1)
    if video.get("expected_frame_count") != frames or video.get("size") != [960, 540]:
        raise ValueError("只接受报告中实际/预期帧数一致的 960×540 原片")
    fps = float(video["actual_fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("报告 FPS 无效")
    expected = supplied_hash or video.get("sha256")
    if expected is None:
        raise ValueError("报告未含 video.sha256；请用 --expected-video-sha256 提供已核对的原片 SHA256")
    expected = digest_value(expected, "expected_video_sha256")
    if video.get("sha256") is not None and digest_value(video["sha256"], "video.sha256") != expected:
        raise ValueError("命令行 SHA256 与报告 video.sha256 不一致")
    actual = sha256(video_path)
    if actual != expected:
        raise ValueError("原片 SHA256 与预期不一致，拒绝加工")
    if video_path.stat().st_size != video["file_size_bytes"]:
        raise ValueError("原片字节数与报告不一致")
    integer(report["global_step"], "global_step")
    integer(report["update_idx"], "update_idx")
    source = report["source"]
    metadata = source["original_metadata"]
    if source.get("kind") != "trained_policy" or str(metadata.get("algo", "")).lower() != "fastsac":
        raise ValueError("本工具仅处理有真实 source 的 FastSAC 训练策略回放")
    if metadata.get("sim") != "motrixsim" or report["playback"].get("simulator") != "motrixsim":
        raise ValueError("源报告没有确认 MotrixSim 回放")
    if source.get("checkpoint") != report.get("checkpoint"):
        raise ValueError("source 与阶段报告的 checkpoint 不一致")
    if digest_value(source["checkpoint_sha256"], "source.checkpoint_sha256") != digest_value(report["checkpoint_sha256"], "checkpoint_sha256"):
        raise ValueError("source 与阶段报告的 checkpoint SHA256 不一致")
    if not source.get("run_dir") or not metadata.get("env_name"):
        raise ValueError("source 缺少训练 run_dir/env_name")
    evaluation = report["evaluation"]
    command = evaluation["command"]
    if not isinstance(command, list) or len(command) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in command
    ):
        raise ValueError("evaluation.command 必须是三维有限速度命令")
    integer(evaluation["seed"], "evaluation.seed")
    if report["playback"].get("evaluation") != evaluation:
        raise ValueError("回放评价设置与报告 evaluation 不一致")
    started = timestamp(report.get("started_utc"), "started_utc")
    finished = timestamp(report.get("finished_utc"), "finished_utc")
    if finished < started:
        raise ValueError("录制结束时间早于开始时间")
    timestamp(metadata.get("created_at"), "source.original_metadata.created_at")
    return report, {"report_path": str(report_path.resolve()), "report_sha256": report_hash,
                    "video_path": str(video_path.resolve()), "video_sha256": actual,
                    "expected_video_sha256_source": "command_line" if supplied_hash else "report.video.sha256",
                    "frames": frames, "fps": fps}


def inspect_video(path: Path, frames: int, fps: float, size: tuple[int, int]) -> dict:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    decoded = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-xerror",
        "-err_detect", "explode", "-i", str(path), "-map", "0:v:0", "-an",
        "-vsync", "0", "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=max(120, frames / fps * 20))
    counts = re.findall(r"^frame=(\d+)\s*$", decoded.stdout, flags=re.MULTILINE)
    actual_frames = int(counts[-1]) if counts else 0
    actual_fps = float(metadata["fps"])
    if decoded.returncode or decoded.stderr.strip() or "progress=end" not in decoded.stdout:
        raise ValueError(f"视频完整解码失败: {decoded.stderr[-3000:]}")
    if actual_frames != frames or list(metadata["size"]) != list(size) or not math.isclose(actual_fps, fps, abs_tol=0.01):
        raise ValueError(f"视频帧数/FPS/尺寸不符: frames={actual_frames}, fps={actual_fps}, size={metadata['size']}")
    return {"actual_frame_count": actual_frames, "actual_fps": actual_fps, "size": list(size),
            "duration_seconds": actual_frames / actual_fps, "decoded_to_end": True,
            "decoder_returncode": decoded.returncode, "sha256": sha256(path),
            "file_size_bytes": path.stat().st_size}


def fit_text(draw, position, text: str, font_path: Path, size: int, width: int, *, color: str | None = None):
    while size > 11:
        font = ImageFont.truetype(str(font_path), size)
        if draw.textbbox((0, 0), text, font=font)[2] <= width:
            break
        size -= 1
    if draw.textbbox((0, 0), text, font=font)[2] > width:
        raise ValueError(f"标注过长，拒绝截断真实信息: {text}")
    draw.text(position, text, font=font, fill=color or COLORS["text"])


def backdrop(report: dict, font_path: Path, preview: bool) -> Image.Image:
    image = Image.new("RGB", CANVAS, COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.line((20, 23, 20, 52), fill=COLORS["accent"], width=3)
    fit_text(draw, (34, 18), "MotrixSim · MicroDuck 强化学习行走", font_path, 28, 1000)
    badge = "样式预览 · 教学标注版" if preview else "教学标注版"
    fit_text(draw, (1050, 28), badge, font_path, 17, 210, color=COLORS["accent"])
    fit_text(draw, (20, 59), "真实策略回放 / 固定评价命令 / 观察步态变化", font_path, 16, 960, color=COLORS["muted"])
    # This area is reserved exclusively for the unscaled original video.
    draw.rectangle((20, 88, 979, 627), fill="black")
    draw.rectangle((1000, 88, 1259, 627), fill=COLORS["panel"])
    draw.line((1000, 88, 1259, 88), fill=COLORS["accent"], width=2)
    fit_text(draw, (1018, 105), "策略与观察", font_path, 18, 224)
    draw.line((1018, 140, 1242, 140), fill=COLORS["line"], width=1)

    def line(y: int, label: str, value: str, *, large: bool = False):
        fit_text(draw, (1018, y), label, font_path, 13, 224, color=COLORS["muted"])
        fit_text(draw, (1018, y + 20), value, font_path, 27 if large else 18, 224)

    line(155, "强化学习算法", "FastSAC")
    line(213, "训练采样迭代", str(report["global_step"]), large=True)
    line(281, "梯度更新计数", str(report["update_idx"]), large=True)
    command = report["evaluation"]["command"]
    line(349, "评价命令 · 非实测速度", f"vx {command[0]:g} m/s")
    fit_text(draw, (1018, 396), f"vy {command[1]:g} m/s  |  wz {command[2]:g} rad/s", font_path, 14, 224, color=COLORS["muted"])
    draw.line((1018, 433, 1242, 433), fill=COLORS["line"], width=1)
    fit_text(draw, (1018, 451), "看三个现象", font_path, 18, 224)
    for y, point in zip((491, 531, 571), OBSERVATION_POINTS):
        fit_text(draw, (1018, y), point, font_path, 16, 224, color=COLORS["muted"])
    fit_text(draw, (20, 648), OBSERVATION_PROMPT, font_path, 20, 1240)
    # A rendered annotation never certifies walking quality. Keep the source
    # acceptance flag in the manifest, and invite observation in the artwork.
    fit_text(draw, (20, 684), WALKING_PROMPT, font_path, 16, 1240, color=COLORS["accent"])
    return image


def decoded_frame(path: Path, index: int) -> Image.Image:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", output_params=["-vsync", "0"])
    try:
        metadata = next(reader)
        for current, raw in enumerate(reader):
            if current == index:
                return Image.frombytes("RGB", tuple(metadata["size"]), raw)
    finally:
        reader.close()
    raise ValueError(f"未能解码 poster 帧 {index}")


def run(args) -> dict:
    stage = args.stage_dir.resolve()
    output = args.output_dir.resolve()
    if not args.font.is_file():
        raise ValueError(f"未找到中文字体: {args.font}")
    report, inputs = load_report(stage, args.expected_video_sha256)
    if output.exists() or output.is_symlink():
        raise ValueError(f"输出目录必须不存在，以保留既有成品: {output}")
    if output == stage or output in stage.parents:
        raise ValueError("输出目录不能覆盖或包含原始阶段目录")
    poster_index = args.poster_frame if args.poster_frame is not None else inputs["frames"] // 2
    if poster_index < 0 or poster_index >= inputs["frames"]:
        raise ValueError("poster-frame 必须位于实际视频帧范围内（从0开始）")
    source_video = Path(inputs["video_path"])
    # Decode-check the complete original again before publishing a derivative.
    original_inspection = inspect_video(source_video, inputs["frames"], inputs["fps"], (960, 540))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "status": "running", "success": False,
                "created_utc": utc_now(), "artifact_kind": "style_preview" if args.preview else "teaching_annotated_recording",
                "teaching_annotation": True, "native_ui_capture": False,
                "training_executed": False, "simulation_executed": False,
                "walking_verified": report.get("walking_verified") is True,
                "inputs": inputs, "original_video_inspection": original_inspection,
                "source": report["source"], "global_step": report["global_step"], "update_idx": report["update_idx"],
                "evaluation": report["evaluation"], "stage_started_utc": report["started_utc"],
                "stage_finished_utc": report["finished_utc"],
                "classroom_annotations": {"walking_prompt": WALKING_PROMPT,
                                          "observation_prompt": OBSERVATION_PROMPT,
                                          "observation_points": list(OBSERVATION_POINTS)},
                "layout": {"canvas": list(CANVAS), "original_video_xywh": list(VIEW),
                           "font_path": str(args.font.resolve()), "font_sha256": sha256(args.font),
                           "style": "dark_gray_light_text_thin_blue_accent", "official_logo_embedded": False},
                "outputs": {}}
    try:
        background = backdrop(report, args.font, args.preview)
        background_path = output / "backdrop.png"
        background.save(background_path)
        output_video = output / "teaching.mp4"
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-v", "error", "-xerror", "-n",
                   "-i", str(source_video), "-loop", "1", "-framerate", f"{inputs['fps']:.12g}", "-i", str(background_path),
                   "-filter_complex", "[1:v][0:v]overlay=x=20:y=88:shortest=1:eof_action=endall,format=yuv420p[v]",
                   "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-vsync", "0",
                   "-movflags", "+faststart", str(output_video)]
        manifest["ffmpeg_command"] = command
        encoded = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=max(120, inputs["frames"] / inputs["fps"] * 30))
        if encoded.returncode or encoded.stderr.strip():
            raise ValueError(f"FFmpeg 教学标注编码失败: {encoded.stderr[-6000:]}")
        manifest["outputs"]["video"] = {"path": str(output_video), **inspect_video(output_video, inputs["frames"], inputs["fps"], CANVAS)}
        frame = decoded_frame(source_video, poster_index)
        poster = background.copy()
        poster.paste(frame, VIEW[:2])
        poster_path = output / "poster.png"
        poster.save(poster_path)
        with Image.open(poster_path) as saved:
            saved.load()
            if saved.size != CANVAS or ImageChops.difference(saved.crop((20, 88, 980, 628)).convert("RGB"), frame).getbbox() is not None:
                raise ValueError("poster 未完整保留原始视频帧")
        manifest["outputs"]["poster"] = {"path": str(poster_path), "size": list(CANVAS), "sha256": sha256(poster_path),
                                            "source": "actual_decoded_original_video_frame", "frame_index_zero_based": poster_index,
                                            "frame_time_seconds": poster_index / inputs["fps"], "original_frame_pixels_preserved": True}
        manifest["outputs"]["backdrop"] = {"path": str(background_path), "sha256": sha256(background_path)}
        manifest.update(status="completed", success=True)
    except BaseException as exc:
        manifest.update(status="failed", success=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            after = {"report_sha256": sha256(Path(inputs["report_path"])), "video_sha256": sha256(source_video)}
            unchanged = all(after[key] == inputs[key] for key in after)
            manifest["input_hashes_after"] = after
            manifest["input_files_unchanged_after_processing"] = unchanged
        except Exception as exc:
            unchanged = False
            manifest["input_hash_guard_error"] = f"{type(exc).__name__}: {exc}"
        if not unchanged:
            manifest.update(status="failed", success=False, error="输入report或原片在加工期间变化；拒绝标记成品成功")
        manifest["finished_utc"] = utc_now()
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not manifest["success"]:
        raise ValueError(manifest["error"])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True, help="已验证的 report.json 与 walking.mp4 所在目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="尚不存在的新输出目录")
    parser.add_argument("--expected-video-sha256", help="报告无 video.sha256 时必须显式提供")
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--poster-frame", type=int, help="poster来源帧，从0开始；默认实际视频中间帧")
    parser.add_argument("--preview", action="store_true", help="显式标记为样式预览，不作为最终课堂成品")
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"ERROR: {type(exc).__name__}: {exc}\n")
    print(json.dumps({"success": True, "output_dir": str(args.output_dir.resolve()), "kind": result["artifact_kind"],
                      "walking_verified": result["walking_verified"], "outputs": result["outputs"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
