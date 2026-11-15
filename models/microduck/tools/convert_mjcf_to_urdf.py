"""Reproducible converter for the pinned MotrixLab MicroDuck, not general MJCF.

Requires numpy, scipy and PyYAML. Native MJCF remains the course runtime model.
An explicit export velocity is required because the source has no velocity limit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1ddb9743c220c3a1e234ea0237d5ce9cc56d0499"
STAND = [0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
         0.3490658503988659, 0.3490658503988659, 0, 0, 0,
         0.08726646259971647, 0.457924, 0.004940, -0.452984]


def vec(text, default):
    return np.array([float(x) for x in (text or default).split()], dtype=float)


def fmt(values):
    return " ".join(f"{float(x):.16g}" for x in values)


def orientation(node):
    unexpected = set(node.attrib) & {"euler", "axisangle", "xyaxes", "zaxis"}
    if unexpected:
        raise ValueError(f"Unsupported orientation on {node.tag}: {unexpected}")
    w, x, y, z = vec(node.get("quat"), "1 0 0 0")
    return Rotation.from_quat([x, y, z, w])


def origin(parent, xyz, rotation=None):
    rot = rotation or Rotation.identity()
    ET.SubElement(parent, "origin", xyz=fmt(xyz), rpy=fmt(rot.as_euler("xyz")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-velocity-rad-s", required=True, type=float,
                        help="URDF parser field only; NOT a hardware or safety limit")
    args = parser.parse_args()
    if not math.isfinite(args.export_velocity_rad_s) or args.export_velocity_rad_s <= 0:
        parser.error("export velocity must be finite and positive")
    source = ET.parse(ROOT / "mjcf/microduck.xml").getroot()
    compiler = source.find("compiler")
    if compiler.get("angle") != "radian" or compiler.get("meshdir") != "assets":
        raise ValueError("This converter expects the pinned model's radian/assets compiler")
    meshes = {m.get("name", Path(m.get("file")).stem): m for m in source.findall("asset/mesh")}
    materials = {m.get("name"): m.get("rgba") for m in source.findall("asset/material")}
    act_defaults = source.find("default/default[@class='chosen_actuator']")
    joint_defaults = act_defaults.find("joint")
    actuator_defaults = act_defaults.find("position")
    effort = max(abs(float(x)) for x in actuator_defaults.get("forcerange").split())
    robot = ET.Element("robot", name="microduck")
    robot.append(ET.Comment("Generated from pinned MotrixLab MJCF. See CONVERSION_REPORT.md for losses."))
    # MuJoCo's URDF extension only controls loading/validation. No actuators implied.
    mj_extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(mj_extension, "compiler", meshdir="../mjcf/assets", strippath="true",
                  discardvisual="false", fusestatic="false")
    for name, rgba in materials.items():
        mat = ET.SubElement(robot, "material", name=name)
        ET.SubElement(mat, "color", rgba=rgba)
    joint_rows, geom_rows = [], []
    world_bodies = source.findall("worldbody/body")
    if len(world_bodies) != 1 or world_bodies[0].get("name") != "trunk_base":
        raise ValueError("Expected single trunk_base root")

    def add_geometry(link, geom, number):
        geom_type = geom.get("type", "mesh")
        cls = geom.get("class")
        if cls not in {"visual", "collision", "self_collision_only"}:
            raise ValueError(f"Unhandled geom class: {cls}")
        kind = "visual" if cls == "visual" else "collision"
        name = geom.get("name", f"{link.get('name')}_{kind}_{number:02d}")
        pos, rot = vec(geom.get("pos"), "0 0 0"), orientation(geom)
        target_names = []
        if geom_type == "mesh":
            item = ET.SubElement(link, kind, name=name)
            origin(item, pos, rot)
            geometry = ET.SubElement(item, "geometry")
            mesh = meshes[geom.get("mesh")]
            if set(mesh.attrib) - {"name", "file", "scale"}:
                raise ValueError("Unhandled mesh transform")
            ET.SubElement(geometry, "mesh", filename="../mjcf/assets/" + mesh.get("file"),
                          scale=mesh.get("scale", "1 1 1"))
            if kind == "visual" and geom.get("material"):
                ET.SubElement(item, "material", name=geom.get("material"))
            target_names.append(name)
        elif geom_type == "capsule" and kind == "collision" and geom.get("fromto"):
            # URDF has no capsule primitive. The union is exact; contact multiplicity is not.
            endpoints = vec(geom.get("fromto"), "").reshape(2, 3)
            radius = float(geom.get("size"))
            delta = endpoints[1] - endpoints[0]
            length = np.linalg.norm(delta)
            if length <= 0:
                raise ValueError("Degenerate capsule")
            direction = delta / length
            pitch = math.acos(np.clip(direction[2], -1, 1))
            yaw = math.atan2(direction[1], direction[0])
            cylinder_rotation = Rotation.from_euler("xyz", [0, pitch, yaw])
            for suffix, xyz, shape in [
                ("cylinder", endpoints.mean(axis=0), "cylinder"),
                ("sphere_a", endpoints[0], "sphere"),
                ("sphere_b", endpoints[1], "sphere"),
            ]:
                target_name = name + "_" + suffix
                item = ET.SubElement(link, kind, name=target_name)
                origin(item, xyz, cylinder_rotation if shape == "cylinder" else None)
                geometry = ET.SubElement(item, "geometry")
                attrs = {"radius": str(radius)}
                if shape == "cylinder":
                    attrs["length"] = str(length)
                ET.SubElement(geometry, shape, **attrs)
                target_names.append(target_name)
        else:
            raise ValueError(f"Unsupported geometry: {geom.attrib}")
        geom_rows.append({"link": link.get("name"), "source_name": geom.get("name"),
                          "source_class": cls, "source_type": geom_type,
                          "urdf_kind": kind, "urdf_names": target_names})

    def visit(body, parent=None):
        name = body.get("name")
        link = ET.SubElement(robot, "link", name=name)
        inertial = body.find("inertial")
        if inertial is None or "fullinertia" not in inertial.attrib:
            raise ValueError(f"Explicit full inertia is required for {name}")
        # MJCF fullinertia is expressed in the body frame; this model uses no inertial rotation.
        if set(inertial.attrib) - {"pos", "mass", "fullinertia"}:
            raise ValueError(f"Unhandled inertia rotation: {name}")
        target_inertial = ET.SubElement(link, "inertial")
        origin(target_inertial, vec(inertial.get("pos"), "0 0 0"))
        ET.SubElement(target_inertial, "mass", value=inertial.get("mass"))
        ixx, iyy, izz, ixy, ixz, iyz = inertial.get("fullinertia").split()
        ET.SubElement(target_inertial, "inertia", ixx=ixx, iyy=iyy, izz=izz,
                      ixy=ixy, ixz=ixz, iyz=iyz)
        for index, geom in enumerate(body.findall("geom")):
            add_geometry(link, geom, index)
        joints = body.findall("joint")
        if parent is None:
            if joints or len(body.findall("freejoint")) != 1:
                raise ValueError("Expected free root")
        else:
            if len(joints) != 1 or body.findall("freejoint"):
                raise ValueError("Expected one hinge per non-root body")
            j = joints[0]
            if j.get("type") != "hinge" or np.any(vec(j.get("pos"), "0 0 0")):
                raise ValueError("Only zero-anchor hinges are supported")
            if j.get("class") != "chosen_actuator" or j.get("ref", "0") != "0":
                raise ValueError("Unexpected joint class/reference")
            joint = ET.SubElement(robot, "joint", name=j.get("name"), type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=name)
            origin(joint, vec(body.get("pos"), "0 0 0"), orientation(body))
            ET.SubElement(joint, "axis", xyz=j.get("axis", "0 0 1"))
            lower, upper = j.get("range").split()
            ET.SubElement(joint, "limit", lower=lower, upper=upper, effort=str(effort),
                          velocity=str(args.export_velocity_rad_s))
            ET.SubElement(joint, "dynamics", damping=joint_defaults.get("damping"),
                          friction=joint_defaults.get("frictionloss"))
            joint_rows.append({"name": j.get("name"), "parent_link": parent, "child_link": name,
                               "axis_local": vec(j.get("axis"), "0 0 1").tolist(),
                               "lower_rad": float(lower), "upper_rad": float(upper),
                               "effort_nm": effort})
        for child in body.findall("body"):
            visit(child, name)

    visit(world_bodies[0])
    order = [a.get("joint") for a in source.findall("actuator/position")]
    if order != [j["name"] for j in joint_rows] or len(order) != len(STAND):
        raise ValueError("Source tree/actuator/default pose order mismatch")
    for index, row in enumerate(joint_rows):
        row.update({"policy_index_zero_based": index, "default_rad": STAND[index]})
    output = ROOT / "urdf/microduck.urdf"
    output.parent.mkdir(exist_ok=True)
    ET.indent(robot, "  ")
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True)
    mapping = {
        "model": "MotrixLab MicroDuck", "source_commit": COMMIT,
        "units": {"length": "m", "angle": "rad", "mass": "kg", "inertia": "kg*m^2"},
        "root_link": "trunk_base", "root_is_free_in_native_mjcf": True,
        "native_root_initial_position_m": vec(world_bodies[0].get("pos"), "0 0 0").tolist(),
        "feet": {"left": "ankle_left", "right": "ankle_right"},
        "policy_joint_order": order, "default_pose_name": "STAND (MotrixLab default)",
        "default_pose_rad": STAND,
        "urdf_velocity_export_field_rad_s": args.export_velocity_rad_s,
        "velocity_field_note": "URDF format placeholder, NOT hardware specification; not used by the course trainer.",
        "source_actuator": {"type": "position", "kp": 0.55, "kv": 0.0,
                            "force_range_nm": [-0.96, 0.96], "armature_kg_m2": 0.0018},
        "joints": joint_rows,
    }
    (ROOT / "joint_map.yaml").write_text(yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (ROOT / "tools/conversion_details.json").write_text(json.dumps({
        "source_commit": COMMIT, "links": len(robot.findall("link")),
        "joints": len(robot.findall("joint")), "geometries": geom_rows,
        "export_velocity_rad_s": args.export_velocity_rad_s,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}: {len(robot.findall('link'))} links, {len(order)} joints")


if __name__ == "__main__":
    main()
