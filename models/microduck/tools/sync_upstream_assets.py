"""Replace only verified matching LFS pointers in the pinned MotrixLab checkout.

Default is a dry run. All files are checked before any --apply write.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    model_dir = args.upstream_root.resolve() / "motrix_robots/src/motrix_robots/assets/microduck"
    if not model_dir.is_dir():
        parser.error("Not a MotrixLab checkout with the expected MicroDuck asset directory")
    if (model_dir / "microduck.xml").read_bytes() != (ROOT / "mjcf/microduck.xml").read_bytes():
        raise ValueError("Model XML differs from the pinned canonical source; no files touched")
    rows = json.loads((ROOT / "tools/lfs_fetch_manifest.json").read_text(encoding="utf-8"))
    actions = []
    for row in rows:
        source = ROOT / row["path"]
        raw = source.read_bytes()
        expected_sha = row["lfs_sha256"]
        if len(raw) != row["size"] or sha256(raw).hexdigest() != expected_sha:
            raise ValueError(f"Bad source asset {source.name}")
        target = model_dir / "assets" / source.name
        if not target.is_file():
            raise ValueError(f"Expected existing LFS pointer/asset is missing: {target.name}")
        existing = target.read_bytes()
        if sha256(existing).hexdigest() == expected_sha:
            continue
        if len(existing) > 1024 or not existing.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            raise ValueError(f"Refusing to overwrite nonmatching non-pointer file: {target.name}")
        lines = existing.decode("ascii").splitlines()
        if f"oid sha256:{expected_sha}" not in lines or f"size {row['size']}" not in lines:
            raise ValueError(f"LFS pointer does not match pinned source: {target.name}")
        actions.append((source, target))
    if args.apply:
        for source, target in actions:
            shutil.copyfile(source, target)
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "verified_total": len(rows),
                      "replaced_or_would_replace": len(actions),
                      "files": [target.name for _, target in actions]}, indent=2))


if __name__ == "__main__":
    main()
