"""Write a reproducible file manifest and optionally verify fixed upstream bytes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1ddb9743c220c3a1e234ea0237d5ce9cc56d0499"
POLLEN_COMMIT = "bae39f5900b318f7e87fc485c30a31213cfa14eb"
MODEL_PATH = "motrix_robots/src/motrix_robots/assets/microduck/"
RAW = f"https://raw.githubusercontent.com/Motphys/MotrixLab/{COMMIT}/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-remote", action="store_true")
    args = parser.parse_args()
    source_files = {
        "mjcf/microduck.xml": RAW + MODEL_PATH + "microduck.xml",
        "mjcf/README.md": RAW + MODEL_PATH + "README.md",
        "licenses/LICENSE-APACHE-2.0.txt": RAW + "LICENSE",
        "licenses/POLLEN-LICENSE-MODELS.md": f"https://raw.githubusercontent.com/pollen-robotics/microduck_rl/{POLLEN_COMMIT}/LICENSE-MODELS.md",
    }
    meshes = {r["path"]: r for r in json.loads((ROOT / "tools/lfs_fetch_manifest.json").read_text(encoding="utf-8"))}
    rows = []
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not set(p.relative_to(ROOT).parts) & {".validation-venv", "__pycache__", ".git"}
             and p.name != "asset_manifest.json"]
    for path in sorted(files):
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        row = {"path": relative, "sha256": sha256(data).hexdigest(), "size": len(data)}
        if relative in meshes:
            source = meshes[relative]
            assert row["sha256"] == source["lfs_sha256"] and len(data) == source["size"]
            row.update({"origin": "MotrixLab fixed-commit Git LFS object", "commit": COMMIT,
                        "source_url": source["source_url"], "license": "CC-BY-NC-SA-4.0",
                        "lfs_sha256": source["lfs_sha256"], "unchanged_from_source": True})
        elif relative in source_files:
            row.update({"source_url": source_files[relative], "unchanged_from_source": True})
            if relative == "licenses/POLLEN-LICENSE-MODELS.md":
                row["commit"] = POLLEN_COMMIT
            else:
                row["commit"] = COMMIT
            row["license"] = "license-notice" if relative.startswith("licenses/") else "Apache-2.0"
            if args.verify_remote:
                with urllib.request.urlopen(row["source_url"], timeout=90) as response:
                    remote = response.read()
                assert remote == data, f"Pinned upstream differs: {relative}"
                row["fixed_remote_bytes_verified"] = True
        else:
            row.update({"origin": "generated teaching model package", "license": "Apache-2.0"})
        rows.append(row)
    report = {"canonical_model": "MotrixLab MicroDuck", "source_commit": COMMIT,
              "source_model_path": MODEL_PATH + "microduck.xml", "supplier_variant": "not selected",
              "generated_urdf_is_not_native_training_model": True,
              "files": rows, "asset_files": len(meshes), "asset_bytes": sum(r["size"] for r in meshes.values()),
              "excluded": ["tools/.validation-venv", "__pycache__", "asset_manifest.json (self)"]}
    (ROOT / "asset_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(rows), "meshes": len(meshes), "remote_source_bytes_verified": args.verify_remote}))


if __name__ == "__main__":
    main()
