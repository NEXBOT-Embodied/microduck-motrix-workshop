#!/bin/sh
# Teacher build-time helper. Not executed automatically; never use during class.
set -eu

course_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
if [ "${1:-}" != "--apply" ] || [ "$#" -ne 1 ]; then
    echo 'Usage: sh image/prepare_linux.sh --apply'
    echo 'Run only in the teacher Linux build instance after the cloud/GPU format is confirmed.'
    exit 2
fi
if [ "$(uname -s)" != Linux ]; then
    echo 'This helper targets Linux; see image/README.md for manual platform preparation.' >&2
    exit 2
fi
command -v uv >/dev/null 2>&1 || { echo 'Prepare uv in the build instance first.' >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { echo 'NVIDIA runtime is not available; choose the correct upstream GPU profile manually.' >&2; exit 2; }

cd "$course_root/upstream/MotrixLab"
sh install.sh --python 3.12 --gpu cuda
runtime_python="$course_root/upstream/MotrixLab/.venv/bin/python"
cd "$course_root"
uv pip install --python "$runtime_python" -e .
uv pip freeze --python "$runtime_python" > image/runtime-freeze.txt
"$runtime_python" -m microduck_course check --runtime
# The check deliberately exits nonzero until real reference/starter policies are registered.
