"""Isolated runs and metadata-backed, hashed checkpoint handling."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from microduck_course.config import BASE_TASK, COURSE_TASK, CourseError, read_yaml, resolve_path, validate_student


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_run(root: Path, settings: dict, student: str) -> tuple[Path, dict]:
    validate_student(student)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    path = resolve_path(root, settings["paths"]["outputs"]) / student / run_id
    path.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "run_id": run_id, "student": student, "status": "running", "upstream_commit": settings["course"]["upstream_commit"], "simulator": "motrixsim", "walking_verified": False}
    write_json(path / "manifest.json", manifest)
    write_json(path / "course_config.json", settings)
    return path, manifest


def open_course_run(root: Path, settings: dict, student: str, run_id: str) -> tuple[Path, dict]:
    validate_student(student)
    if not re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{8}", run_id):
        raise CourseError("--run 必须是当前学员训练输出的 run_id。")
    path = resolve_path(root, settings["paths"]["outputs"]) / student / run_id
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourseError(f"找不到有效运行记录: {path}") from exc
    if manifest.get("status") != "completed":
        raise CourseError(f"运行状态为 {manifest.get('status')}，不能作为已训练策略回放或导出。")
    return path, manifest


def validate_policy_files(run_dir: Path, checkpoint: Path, expected_hash: str) -> dict:
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise CourseError(f"缺少非空真实策略: {checkpoint}")
    try:
        checkpoint.resolve().relative_to(run_dir.resolve())
    except ValueError as exc:
        raise CourseError("checkpoint 必须位于其 metadata-backed run_dir 内。") from exc
    try:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourseError(f"缺少有效 metadata.json: {run_dir}") from exc
    if metadata.get("env_name") not in (BASE_TASK, COURSE_TASK) or metadata.get("rllib") != "motrix" or metadata.get("algo") != "fastsac":
        raise CourseError("策略必须来自官方 MicroDuck 或本课程任务的 motrix.fastsac 训练。")
    if metadata.get("sim") not in (None, "motrixsim"):
        raise CourseError("策略源运行没有使用 MotrixSim。")
    read_yaml(run_dir / "task_config.yaml")
    if not re.fullmatch(r"[a-f0-9]{64}", str(expected_hash)) or sha256(checkpoint) != expected_hash:
        raise CourseError(f"checkpoint SHA256 不匹配或未登记: {checkpoint}")
    return metadata


def resolve_registered_policy(root: Path, settings: dict, role: str) -> tuple[Path, Path, dict]:
    entry = read_yaml(resolve_path(root, settings["paths"]["policies"]))[role]
    if entry.get("status") != "ready" or not entry.get("verified_walking"):
        raise CourseError(f"{role} 策略尚未完成真实行走验收；教师需补齐 configs/policies.yaml。")
    if not entry.get("run_dir") or not entry.get("checkpoint"):
        raise CourseError(f"{role} 策略路径未登记。")
    run_dir = resolve_path(root, entry["run_dir"])
    checkpoint = resolve_path(run_dir, entry["checkpoint"])
    metadata = validate_policy_files(run_dir, checkpoint, entry.get("sha256"))
    return run_dir, checkpoint, metadata


def policy_provenance(run_dir: Path, checkpoint: Path, role: str) -> dict:
    """Capture the starting artifact identity before any trainer is invoked."""
    metadata_path = run_dir / "metadata.json"
    return {
        "kind": {"starter": "teacher_starter", "reference": "teacher_reference"}.get(role, "trained_policy"),
        "role": role, "run_dir": str(run_dir.resolve()),
        "checkpoint": checkpoint.resolve().relative_to(run_dir.resolve()).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "original_metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "metadata_sha256": sha256(metadata_path),
        "task_config_sha256": sha256(run_dir / "task_config.yaml"),
    }


def validate_export_selection(selected: Path, expected: Path, expected_hash: str) -> None:
    if selected.resolve() != expected.resolve() or sha256(selected) != expected_hash:
        raise CourseError("官方 best policy 指针/文件已变化，与本次 manifest 策略不同；拒绝导出。")


def course_run_policy(path: Path, manifest: dict) -> tuple[Path, Path, dict]:
    run_dir = resolve_path(path, manifest["upstream_run_dir"])
    checkpoint = resolve_path(run_dir, manifest["checkpoint"])
    metadata = validate_policy_files(run_dir, checkpoint, manifest["checkpoint_sha256"])
    return run_dir, checkpoint, metadata
