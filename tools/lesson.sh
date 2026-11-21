#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${COURSE_PYTHON:-}" ]]; then
    python="$COURSE_PYTHON"
elif [[ -x "$root/.venv/bin/python" ]]; then
    python="$root/.venv/bin/python"
else
    python="$root/upstream/MotrixLab/.venv/bin/python"
fi
if [[ ! -x "$python" ]]; then
    echo "Prepared Python not found. Run bash tools/install_course.sh, or set COURSE_PYTHON to an absolute interpreter path." >&2
    exit 2
fi
export PYTHONUTF8=1
export MICRODUCK_LESSON_BRIEF=1
cd -- "$root"
exec "$python" "$root/tools/local_tutorial.py" "$@"
