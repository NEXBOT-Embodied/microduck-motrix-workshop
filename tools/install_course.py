"""Install the locked course dependencies using the selected download indexes."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib


PROJECT = Path(__file__).resolve().parents[1]
MOTRIXLAB = PROJECT / "upstream" / "MotrixLab"
PYPI = "https://pypi.org/simple"
PYPI_MIRROR = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
MOTPHYS = "https://pypi.motphys.com/simple"
PYTORCH = "https://download.pytorch.org/whl"
PYTORCH_MIRROR = "https://mirrors.nju.edu.cn/pytorch/whl"
CPU_WHEELS = {
    "win32": "69e25c973bdd7ea24b0fa9f9792114950afaeb8f819e5723819b923f50989175",
    "linux": "64123c05615e27368c7a7816f6e39c6d219998693beabde0b0b9cedf91b5ed8b",
}
REQUIREMENT = re.compile(r"^([a-z0-9-]+)==([^ ;\\]+)")


def install(uv: str, backend: str, upstream: bool) -> None:
    pypi = PYPI if upstream else PYPI_MIRROR
    pytorch = PYTORCH if upstream else PYTORCH_MIRROR
    venv = Path(os.environ.get("UV_PROJECT_ENVIRONMENT") or MOTRIXLAB / ".venv").resolve()
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def run(*args, capture=False):
        return subprocess.run(
            [uv, "--no-config", *map(str, args)], cwd=MOTRIXLAB, check=True, text=True,
            encoding="utf-8", stdout=subprocess.PIPE if capture else None,
        ).stdout

    print(f"Course backend: {backend}\nEnvironment: {venv}", flush=True)
    if not python.exists():
        run("venv", "--no-python-downloads", "--python", sys.executable, venv)

    # The CUDA recipe supplies the common runtime dependencies, including
    # those shared by PyTorch's 2.7.0 CPU wheel. The CPU profile omits the
    # GPU-only libraries and installs that separately pinned wheel.
    export = ("export", "--frozen", "--no-default-groups", "--extra", "cuda")
    external = run(*export, "--no-emit-workspace", capture=True)
    workspace = run(*export, "--only-emit-workspace", capture=True)
    with (MOTRIXLAB / "uv.lock").open("rb") as stream:
        locked = tomllib.load(stream)
    sources = {
        (pkg["name"], pkg["version"]): pkg["source"]["registry"].rstrip("/")
        for pkg in locked["package"] if "registry" in pkg["source"]
    }
    indexes = {PYPI: pypi, MOTPHYS: MOTPHYS, PYTORCH + "/cu128": pytorch + "/cu128"}
    requirements = {index: [] for index in indexes.values()}
    for entry in re.split(r"(?m)(?=^[a-z0-9-]+==)", external):
        if not (match := REQUIREMENT.match(entry)):
            continue
        name, version = match.groups()
        # Universal exports can retain PyPI Torch's CUDA 13 dependencies even
        # when the workspace CUDA extra was selected. Only keep the selected
        # Torch wheel family before resolving the pinned files per index.
        if name in {"torch", "torchaudio", "torchvision"} and (
            backend == "cpu" or not version.endswith("+cu128")
        ):
            continue
        if name == "triton" and (backend == "cpu" or version != "3.3.0"):
            continue
        if name.startswith("cuda-") or (
            name.startswith("nvidia-") and name != "nvidia-ml-py"
            and (backend == "cpu" or not name.endswith("-cu12"))
        ):
            continue
        try:
            index = indexes[sources[name, version]]
        except KeyError as exc:
            raise ValueError(f"Unexpected registry for {name}=={version}; inspect the MotrixLab lock.") from exc
        requirements[index].append(entry)
    if backend == "cpu":
        requirements[pytorch + "/cpu"] = [
            f"torch==2.7.0+cpu --hash=sha256:{CPU_WHEELS[sys.platform]}\n"
        ]

    pip = ("pip", "install", "--python", python, "--no-deps")
    with tempfile.TemporaryDirectory() as tmp:
        for index, entries in requirements.items():
            if not entries:
                continue
            path = Path(tmp) / "requirements.txt"
            path.write_text("".join(entries), encoding="utf-8")
            print(f"Installing locked dependencies from {index}", flush=True)
            run(*pip, "--require-hashes", "--default-index", index, "--requirement", path)
    local = [part for line in workspace.splitlines() if line.startswith("-e ") for part in ("--editable", line[3:])]
    if not local:
        raise ValueError("MotrixLab workspace export did not include its local packages.")
    run(*pip, "--default-index", pypi, *local, "--editable", PROJECT)
    run("pip", "check", "--python", python)
    subprocess.run(
        [python, "-c", "import torch, motrixsim, numba, imageio_ffmpeg; "
         "print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"],
        check=True,
    )
    print("Installed. Run tools/lesson.sh or tools\\lesson.cmd from the course root.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--upstream", action="store_true")
    parser.add_argument("--uv", required=True)
    args = parser.parse_args()
    try:
        install(args.uv, args.backend, args.upstream)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Installation stopped: {error}\n")
