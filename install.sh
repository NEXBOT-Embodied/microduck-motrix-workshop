#!/bin/sh
set -eu

usage() {
    cat <<'EOF'
Usage: sh install.sh [--gpu auto|cuda|default] [--official]

  auto      Use CUDA on Windows and on Linux with an available NVIDIA driver.
            Use the default PyTorch build on Apple Silicon and other Linux PCs.
  cuda      Use the locked CUDA 12.8 profile (Windows or Linux x86_64).
  default   Use the locked PyPI PyTorch build.
  --official  Use official download sources from the start.
EOF
}

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

gpu=auto
official=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu)
            [ "$#" -ge 2 ] || die '--gpu requires auto, cuda or default.'
            gpu=$2
            shift
            ;;
        --official) official=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1. See --help." ;;
    esac
    shift
done

case "$gpu" in
    auto|cuda|default) ;;
    *) die "Unsupported GPU profile: $gpu. See --help." ;;
esac

course_root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
lab_root="$course_root/upstream/MotrixLab"
venv="$lab_root/.venv"
system=$(uname -s)
arch=$(uname -m)
case "$system:$arch" in
    Darwin:arm64)
        uv_platform=aarch64-apple-darwin
        uv_sha256=5aeca67f2283f75235de40d0bd9c0a5bbc5fce23b6d48b3bf9fa1b2db08382ca
        runtime_python="$venv/bin/python"
        [ "$gpu" != cuda ] || die 'CUDA requires Windows or Linux x86_64.'
        ;;
    Linux:x86_64)
        uv_platform=x86_64-unknown-linux-gnu
        uv_sha256=a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5
        runtime_python="$venv/bin/python"
        ;;
    MINGW*:x86_64|MSYS*:x86_64)
        system=Windows
        uv_platform=x86_64-pc-windows-msvc
        uv_sha256=edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb
        runtime_python="$venv/Scripts/python.exe"
        ;;
    *) die "The pinned MotrixSim SDK has no wheel for $system/$arch. Use Apple Silicon macOS or x86_64 Linux/Windows." ;;
esac

if [ "$gpu" = auto ]; then
    gpu=default
    if [ "$system" = Windows ] || {
        [ "$system" = Linux ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
    }; then
        gpu=cuda
    fi
fi

[ -f "$lab_root/uv.lock" ] || die 'MotrixLab uv.lock is missing. Use the complete workshop checkout.'
printf '==> %s/%s, Python 3.12, PyTorch profile: %s\n' "$system" "$arch" "$gpu"
temp_dir=$(mktemp -d "$course_root/.venv-install.XXXXXX")
trap 'find "$temp_dir" -depth -delete' EXIT
trap 'exit 1' HUP INT TERM

pypi_official=https://pypi.org/simple
pypi_mirror=https://mirrors.aliyun.com/pypi/simple
python_official=https://github.com/astral-sh/python-build-standalone/releases/download
python_mirror=https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone
uv_release=0.13.9
uv_bin="$course_root/.venv-tools/uv"
if [ "$system" = Windows ]; then
    uv_bin="$uv_bin.exe"
fi

download_uv() {
    curl --fail --location --silent --show-error --connect-timeout 10 --retry 2 \
        "$1/$uv_archive" --output "$temp_dir/$uv_archive" || return 1
    if [ "$system" = Darwin ]; then
        printf '%s  %s\n' "$uv_sha256" "$uv_archive" | (cd "$temp_dir" && shasum -a 256 -c -)
    else
        printf '%s  %s\n' "$uv_sha256" "$uv_archive" | (cd "$temp_dir" && sha256sum -c -)
    fi
}

if [ ! -x "$uv_bin" ]; then
    if command -v uv >/dev/null 2>&1 && uv --version | awk -F '[ .]+' '
        $2 > 0 || $3 > 13 || ($3 == 13 && $4 >= 9) { supported = 1 }
        END { exit !supported }
    '; then
        uv_bin=$(command -v uv)
    else
        printf '==> Installing uv %s in .venv-tools\n' "$uv_release"
        if [ "$system" = Windows ]; then
            uv_archive="uv-$uv_platform.zip"
        else
            uv_archive="uv-$uv_platform.tar.gz"
        fi
        if [ "$official" -eq 1 ] ||
            ! download_uv "https://mirrors.ustc.edu.cn/github-release/astral-sh/uv/$uv_release"; then
            download_uv "https://github.com/astral-sh/uv/releases/download/$uv_release"
        fi
        if [ "$system" = Windows ]; then
            unzip -q "$temp_dir/$uv_archive" uv.exe -d "$temp_dir"
            downloaded_uv="$temp_dir/uv.exe"
        else
            tar -xzf "$temp_dir/$uv_archive" -C "$temp_dir" "uv-$uv_platform/uv"
            downloaded_uv="$temp_dir/uv-$uv_platform/uv"
        fi
        "$downloaded_uv" --version
        mkdir -p "$(dirname -- "$uv_bin")"
        mv -f "$downloaded_uv" "$uv_bin"
    fi
fi

create_venv() {
    UV_PYTHON_INSTALL_MIRROR="$1" "$uv_bin" --no-config venv --python 3.12 --allow-existing "$venv"
}

if [ -x "$runtime_python" ]; then
    "$runtime_python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' ||
        die "Expected a working Python 3.12 at $runtime_python. Move the existing .venv aside and retry."
elif [ "$official" -eq 1 ] || ! create_venv "$python_mirror"; then
    create_venv "$python_official"
fi

set -- --frozen --no-default-groups --format pylock.toml
[ "$system" != Linux ] || set -- "$@" --extra tbb
[ "$gpu" != cuda ] || set -- "$@" --extra cuda
official_lock="$temp_dir/pylock.official.toml"
"$uv_bin" --directory "$lab_root" export "$@" --output-file "$official_lock" >/dev/null
# The temporary lock lives one level below the course root and can sync the
# workshop editable alongside the upstream workspace. PyYAML is already locked.
printf '\n[[packages]]\nname = "microduck-course"\ndirectory = { path = "..", editable = true }\n' >> "$official_lock"

sync_environment() {
    "$uv_bin" --no-config pip sync --preview-features pylock --strict \
        --python "$runtime_python" --default-index "$1" "$2"
}

if [ "$official" -eq 0 ]; then
    mirror_lock="$temp_dir/pylock.mirrors.toml"
    # Frozen uv locks contain direct artifact URLs. Redirect only sources that
    # mirror those paths byte-for-byte, and retain every original download hash.
    sed \
        -e 's|url = "https://files\.pythonhosted\.org/packages/|url = "https://mirrors.aliyun.com/pypi/packages/|g' \
        -e 's|index = "https://pypi\.org/simple"|index = "https://mirrors.aliyun.com/pypi/simple"|g' \
        -e 's|url = "https://download-r2\.pytorch\.org/whl/cu128/|url = "https://mirrors.nju.edu.cn/pytorch/whl/cu128/|g' \
        -e 's|url = "https://download\.pytorch\.org/whl/cu128/|url = "https://mirrors.nju.edu.cn/pytorch/whl/cu128/|g' \
        -e 's|index = "https://download\.pytorch\.org/whl/cu128"|index = "https://mirrors.nju.edu.cn/pytorch/whl/cu128"|g' \
        "$official_lock" > "$mirror_lock"
    printf '==> Installing locked packages using domestic mirrors and the Motphys SDK index\n'
    if ! sync_environment "$pypi_mirror" "$mirror_lock"; then
        printf '==> Mirror install failed. Retrying with official sources.\n' >&2
        sync_environment "$pypi_official" "$official_lock"
    fi
else
    printf '==> Installing locked packages from official sources\n'
    sync_environment "$pypi_official" "$official_lock"
fi

cd "$course_root"
PYTHONUTF8=1 "$runtime_python" -c 'import motrixsim, torch'
PYTHONUTF8=1 "$runtime_python" tools/local_tutorial.py check --brief
