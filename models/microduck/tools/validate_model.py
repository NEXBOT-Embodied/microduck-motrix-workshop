"""Validate real resources, topology, inertia and FK against the pinned MJCF.

This is NOT a walking-policy test and does not certify contact/dynamic equivalence.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]


def vec(text, default="0 0 0"):
    return np.fromstring(text or default, sep=" ")


def tensor(attrs, style):
    if style == "mjcf":
        xx, yy, zz, xy, xz, yz = vec(attrs["fullinertia"])
    else:
        xx, yy, zz, xy, xz, yz = (float(attrs[k]) for k in ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"])
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def transform(xyz, rotation):
    t = np.eye(4)
    t[:3, :3] = rotation.as_matrix()
    t[:3, 3] = xyz
    return t


def urdf_fk(urdf, q, root_transform):
    frames = {"trunk_base": root_transform}
    pending = list(urdf.findall("joint"))
    while pending:
        changed = False
        for j in list(pending):
            parent = j.find("parent").get("link")
            if parent not in frames:
                continue
            o = j.find("origin")
            local = transform(vec(o.get("xyz")), Rotation.from_euler("xyz", vec(o.get("rpy"))))
            axis = vec(j.find("axis").get("xyz"))
            motion = transform(np.zeros(3), Rotation.from_rotvec(axis * q[j.get("name")]))
            frames[j.find("child").get("link")] = frames[parent] @ local @ motion
            pending.remove(j)
            changed = True
        if not changed:
            raise ValueError("URDF joint tree is disconnected or cyclic")
    return frames


def main():
    report = {"scope": "resources, topology, inertial consistency, loading, static FK; no rollout/training",
              "mujoco_version": mujoco.__version__, "checks": {}}
    native_path = ROOT / "mjcf/microduck.xml"
    urdf_path = ROOT / "urdf/microduck.urdf"
    native = ET.parse(native_path).getroot()
    urdf = ET.parse(urdf_path).getroot()
    mapping = yaml.safe_load((ROOT / "joint_map.yaml").read_text(encoding="utf-8"))
    source_bodies = {b.get("name"): b for b in native.findall(".//worldbody//body")}
    urdf_links = {b.get("name"): b for b in urdf.findall("link")}
    source_joints = {j.get("name"): j for j in native.findall(".//worldbody//joint")}
    urdf_joints = {j.get("name"): j for j in urdf.findall("joint")}
    assert len(source_bodies) == len(urdf_links) == 15
    assert len(source_joints) == len(urdf_joints) == 14
    assert source_bodies.keys() == urdf_links.keys()
    assert source_joints.keys() == urdf_joints.keys()
    assert len({j.find("child").get("link") for j in urdf_joints.values()}) == 14
    assert set(urdf_links) - {j.find("child").get("link") for j in urdf_joints.values()} == {"trunk_base"}
    report["checks"]["topology"] = {"links": 15, "hinges": 14, "root": "trunk_base", "ok": True}

    fetched = json.loads((ROOT / "tools/lfs_fetch_manifest.json").read_text(encoding="utf-8"))
    resource_rows = []
    for row in fetched:
        path = ROOT / row["path"]
        data = path.read_bytes()
        assert sha256(data).hexdigest() == row["lfs_sha256"], row["path"]
        assert len(data) == row["size"] and not data.startswith(b"version https://git-lfs")
        # All canonical assets are binary STL. Length and finite vertex data catch common truncation.
        count = struct.unpack_from("<I", data, 80)[0]
        assert len(data) == 84 + count * 50, row["path"]
        faces = np.frombuffer(data, dtype=np.dtype([
            ("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]), offset=84)
        assert np.isfinite(faces["vertices"]).all()
        resource_rows.append({"path": row["path"], "faces": count, "size": len(data), "ok": True})
    for mesh in urdf.findall(".//geometry/mesh"):
        assert (urdf_path.parent / mesh.get("filename")).resolve().is_file()
    report["checks"]["resources"] = {"ok": True, "files": resource_rows,
                                      "total_bytes": sum(r["size"] for r in resource_rows)}
    assert len(resource_rows) == 40

    inertial_rows = []
    for name, body in source_bodies.items():
        a, b = body.find("inertial"), urdf_links[name].find("inertial")
        ia, ib = tensor(a.attrib, "mjcf"), tensor(b.find("inertia").attrib, "urdf")
        mass = float(a.get("mass"))
        eig = np.linalg.eigvalsh(ia)
        assert mass > 0 and eig.min() > 0
        assert eig.max() <= eig.sum() - eig.max() + 1e-12, name
        assert mass == float(b.find("mass").get("value"))
        assert np.max(np.abs(ia - ib)) < 1e-15
        assert np.max(np.abs(vec(a.get("pos")) - vec(b.find("origin").get("xyz")))) < 1e-14
        inertial_rows.append({"link": name, "mass_kg": mass,
                              "inertia_eigenvalues_kg_m2": eig.tolist(), "ok": True})
    report["checks"]["source_inertia"] = {"ok": True, "links": inertial_rows,
                                           "total_mass_kg": sum(r["mass_kg"] for r in inertial_rows)}
    # MuJoCo 3.3.7's native Windows file loader cannot open this Unicode workspace path.
    # Python reads the exact bytes and supplies meshes through MuJoCo's documented VFS.
    vfs_assets = {p.name: p.read_bytes() for p in (ROOT / "mjcf/assets").glob("*.stl")}
    model = mujoco.MjModel.from_xml_string(native_path.read_text(encoding="utf-8"), assets=vfs_assets)
    imported = mujoco.MjModel.from_xml_string(urdf_path.read_text(encoding="utf-8"), assets=vfs_assets)
    report["checks"]["loading"] = {"ok": True, "resource_loader": "Python file bytes + MuJoCo VFS (Unicode Windows path workaround)",
        "native": {"nbody_including_world": model.nbody, "nq": model.nq, "nv": model.nv,
                   "joints_including_free": model.njnt, "actuators": model.nu, "geoms": model.ngeom},
        "urdf": {"nbody_including_world": imported.nbody, "nq": imported.nq, "nv": imported.nv,
                 "joints": imported.njnt, "actuators": imported.nu, "geoms": imported.ngeom}}
    assert model.njnt == 15 and imported.njnt == 14
    assert model.nu == 14 and imported.nu == 0
    compiled_inertia_rows = []
    for name in source_bodies:
        a, b = model.body(name), imported.body(name)
        qa, qb = a.iquat, b.iquat
        ra = Rotation.from_quat([qa[1], qa[2], qa[3], qa[0]]).as_matrix()
        rb = Rotation.from_quat([qb[1], qb[2], qb[3], qb[0]]).as_matrix()
        ta, tb = ra @ np.diag(a.inertia) @ ra.T, rb @ np.diag(b.inertia) @ rb.T
        errors = {"mass_error_kg": float(np.max(np.abs(a.mass - b.mass))),
                  "com_error_m": float(np.max(np.abs(a.ipos - b.ipos))),
                  "tensor_error_kg_m2": float(np.max(np.abs(ta - tb)))}
        assert errors["mass_error_kg"] < 1e-12 and errors["com_error_m"] < 1e-12
        assert errors["tensor_error_kg_m2"] < 1e-12
        compiled_inertia_rows.append({"link": name, **errors})
    report["checks"]["compiled_inertia"] = {"ok": True, "links": compiled_inertia_rows}

    order = mapping["policy_joint_order"]
    rng = np.random.default_rng(20261206)
    lower = np.array([float(source_joints[j].get("range").split()[0]) for j in order])
    upper = np.array([float(source_joints[j].get("range").split()[1]) for j in order])
    poses = {"zero": np.zeros(14), "STAND": np.array(mapping["default_pose_rad"])}
    for i in range(10):
        poses[f"random_in_range_{i:02}"] = rng.uniform(lower, upper)
    native_data, imported_data = mujoco.MjData(model), mujoco.MjData(imported)
    rows = []
    # Add a nontrivial floating-base rotation/translation to test frame conventions.
    base_rot = Rotation.from_euler("xyz", [0.13, -0.21, 0.37])
    base_position = np.array([0.17, -0.11, 0.32])
    base_quat = base_rot.as_quat()
    for pose_name, values in poses.items():
        mujoco.mj_resetData(model, native_data)
        mujoco.mj_resetData(imported, imported_data)
        native_data.qpos[:3] = base_position
        native_data.qpos[3:7] = [base_quat[3], *base_quat[:3]]
        for name, value in zip(order, values):
            native_data.qpos[model.joint(name).qposadr[0]] = value
            imported_data.qpos[imported.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, native_data)
        mujoco.mj_forward(imported, imported_data)
        root_tf = transform(base_position, base_rot)
        independent_fk = urdf_fk(urdf, dict(zip(order, values)), root_tf)
        imported_root = imported_data.body("trunk_base")
        imported_tf = np.eye(4)
        imported_tf[:3, :3] = imported_root.xmat.reshape(3, 3)
        imported_tf[:3, 3] = imported_root.xpos
        alignment = root_tf @ np.linalg.inv(imported_tf)
        max_position, max_rotation, max_independent = 0., 0., 0.
        for name in source_bodies:
            a, b = native_data.body(name), imported_data.body(name)
            tb = np.eye(4)
            tb[:3, :3], tb[:3, 3] = b.xmat.reshape(3, 3), b.xpos
            tb = alignment @ tb
            max_position = max(max_position, float(np.max(np.abs(a.xpos - tb[:3, 3]))))
            max_rotation = max(max_rotation, float(np.max(np.abs(a.xmat.reshape(3, 3) - tb[:3, :3]))))
            native_tf = np.eye(4)
            native_tf[:3, :3], native_tf[:3, 3] = a.xmat.reshape(3, 3), a.xpos
            max_independent = max(max_independent, float(np.max(np.abs(native_tf - independent_fk[name]))))
        assert max_position < 1e-10 and max_rotation < 1e-10 and max_independent < 1e-10
        rows.append({"pose": pose_name, "position_max_abs_m": max_position,
                     "rotation_matrix_max_abs": max_rotation,
                     "independent_urdf_fk_max_abs": max_independent})
    report["checks"]["forward_kinematics"] = {"ok": True, "poses": rows,
        "compares": "All 15 link frames: native MJCF vs compiled URDF and independent URDF FK; root aligned"}
    report["status"] = "PASS"
    report["limitations"] = [
        "No trained walking policy, reward, physics rollout or MotrixSim execution was tested.",
        "URDF import has a fixed root and no actuators; contact masks, armature and actuator laws differ.",
        "Capsule decomposes into 3 collision primitives; contact counts may differ.",
        "URDF velocity field is an explicit export placeholder, not a measured motor limit."]
    (ROOT / "tools/validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "mesh_files": 40,
                      "total_mass_kg": report["checks"]["source_inertia"]["total_mass_kg"],
                      "fk_poses": len(rows), "max_fk_position_error_m": max(x["position_max_abs_m"] for x in rows),
                      "max_fk_rotation_matrix_error": max(x["rotation_matrix_max_abs"] for x in rows)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise
