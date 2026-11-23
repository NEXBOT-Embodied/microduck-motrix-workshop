#!/bin/sh
set -eu

course_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
if [ -z "${COURSE_PYTHON:-}" ]; then
    COURSE_PYTHON="$course_root/.venv/bin/python"
    if [ ! -x "$COURSE_PYTHON" ]; then
        COURSE_PYTHON="$course_root/upstream/MotrixLab/.venv/bin/python"
    fi
fi
if [ ! -x "$COURSE_PYTHON" ]; then
    printf 'Prepared Python not found: %s\nRun sh install.sh or set COURSE_PYTHON.\n' "$COURSE_PYTHON" >&2
    exit 2
fi
# Resolve an explicitly supplied relative interpreter before changing directories.
case "$COURSE_PYTHON" in
    /*) ;;
    *) COURSE_PYTHON="$PWD/$COURSE_PYTHON" ;;
esac

cd "$course_root"
export PYTHONUTF8=1 MICRODUCK_LESSON_BRIEF=1 MICRODUCK_COURSE_ROOT="$course_root"
exec "$COURSE_PYTHON" tools/local_tutorial.py "$@"
