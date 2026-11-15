"""Restore pinned real LFS assets; never accepts Git LFS pointer text as a mesh."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def restore(row):
    target = ROOT / row["path"]
    if target.exists():
        data = target.read_bytes()
        if len(data) == row["size"] and sha256(data).hexdigest() == row["lfs_sha256"]:
            return row["path"], "already_verified"
    with urllib.request.urlopen(row["source_url"], timeout=120) as response:
        data = response.read()
    if len(data) != row["size"] or sha256(data).hexdigest() != row["lfs_sha256"]:
        raise ValueError(f"LFS hash/size mismatch: {row['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return row["path"], "restored_verified"


if __name__ == "__main__":
    rows = json.loads((ROOT / "tools/lfs_fetch_manifest.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(restore, rows))
    print(json.dumps({"ok": len(results), "files": results}, indent=2, ensure_ascii=False))
