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
command -v nvidia-smi >/dev/null 2>&1 || { echo 'NVIDIA runtime is not available; choose the correct upstream GPU profile manually.' >&2; exit 2; }

sh "$course_root/install.sh" --gpu cuda
runtime_python="$course_root/upstream/MotrixLab/.venv/bin/python"
uv_bin="$course_root/.venv-tools/uv"
if [ ! -x "$uv_bin" ]; then
    uv_bin=$(command -v uv)
fi
cd "$course_root"
"$uv_bin" pip freeze --python "$runtime_python" > image/runtime-freeze.txt
"$runtime_python" -m microduck_course check --runtime
