"""Read-only readiness evidence. Missing runtime does not imply a working image."""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from microduck_course.config import CourseError, resolve_path
from microduck_course.storage import resolve_registered_policy


def inspect_mjcf_assets(model: Path) -> dict:
    if not model.is_file():
        return {"status": "missing_model", "model": str(model)}
    try:
        tree = ET.parse(model)
    except ET.ParseError as exc:
        return {"status": "invalid_xml", "error": str(exc)}
    compiler = tree.getroot().find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    missing, lfs = [], []
    for mesh in tree.findall(".//mesh[@file]"):
        path = model.parent / meshdir / mesh.attrib["file"]
        if not path.is_file():
            missing.append(str(path))
        else:
            with path.open("rb") as file:
                if file.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    lfs.append(str(path))
    return {"status": "assets_present_static_only" if not missing and not lfs else "assets_blocked", "model": str(model), "missing": missing, "lfs_pointers": lfs}


def inspect_readiness(root: Path, settings: dict) -> dict:
    from microduck_course.runtime import RUNTIME_MODULES
    modules = {}
    for name in RUNTIME_MODULES:
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            modules[name] = False
    upstream = resolve_path(root, settings["course"]["upstream_path"])
    model = upstream / "motrix_robots/src/motrix_robots/assets/microduck/microduck.xml"
    policies = {}
    for role in ("reference", "starter"):
        try:
            run_dir, checkpoint, _ = resolve_registered_policy(root, settings, role)
            policies[role] = {"status": "registered_files_valid", "run_dir": str(run_dir), "checkpoint": str(checkpoint), "runtime_load_tested": False}
        except CourseError as exc:
            policies[role] = {"status": "blocked", "reason": str(exc)}
    assets = inspect_mjcf_assets(model)
    blocked = not all(modules.values()) or assets["status"] != "assets_present_static_only" or any(p["status"] == "blocked" for p in policies.values())
    return {
        "status": "blocked" if blocked else "static_checks_passed_runtime_unverified",
        "python": sys.version.split()[0], "modules": modules, "assets": assets,
        "policies": policies, "capacity_50_users": "unverified", "cloud_login_tested": False,
        "motrixsim_render_tested": False, "sim2sim_completed": False,
    }
