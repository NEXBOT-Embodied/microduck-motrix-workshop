#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
backend=cuda
upstream=false
while (($#)); do
    case "$1" in
        --backend)
            if (($# < 2)); then echo '--backend requires cuda or cpu.' >&2; exit 2; fi
            backend="$2"; shift 2 ;;
        --upstream) upstream=true; shift ;;
        -h|--help)
            echo 'Usage: bash tools/install_course.sh [--backend cuda|cpu] [--upstream]'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$backend" in
    cuda|cpu) ;;
    *) echo 'Choose --backend cuda or --backend cpu.' >&2; exit 2 ;;
esac

if [[ "$(uname -m)" != x86_64 ]]; then
    echo 'This course installer supports x86_64 Windows and Ubuntu.' >&2; exit 2
fi
case "$(uname -s)" in
    MINGW*|MSYS*)
        wheel='41/c7/3ad22f0d3f52497bef079ac1a6805c994ca68148bd273d11a61cb5c4bf56/uv-0.11.23-py3-none-win_amd64.whl'
        hash=b3f515fd6b43068f241467496bced62cb2ed36d52d4c0877cfe61a1240713d32
        python_platform=x86_64-pc-windows-msvc
        python_hash=1f2ab53880fc67ea323398c6388e05af28addc99a8e389209f08a51fdff4b149
        python_bin=.venv-course-python/python/python.exe
        uv_name=uv.exe ;;
    Linux)
        wheel='19/ff/764e1c21ba988589d2b505d2b06876b5f06ffe7cc6858dff6cc3faf7cb14/uv-0.11.23-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
        hash=7a85330de0a7eb0d5c6cf03c80edfb86facad19df367a0b52fc906db1ab15ce9
        python_platform=x86_64-unknown-linux-gnu
        python_hash=ce4f6d0a6cfdb3bed6e44c2f9affd85943c76a42e25ad3a259002f31d6fb8a20
        python_bin=.venv-course-python/python/bin/python3.12
        uv_name=uv ;;
    *) echo 'This course installer supports x86_64 Windows and Ubuntu.' >&2; exit 2 ;;
esac

bootstrap="$(mktemp -d)"
trap 'rm -f -- "$bootstrap/uv.whl" "$bootstrap/$uv_name" "$bootstrap/python.tar.gz"; rmdir -- "$bootstrap"' EXIT
download() {
    curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
        --retry 2 --connect-timeout 12 "$1" --output "$2"
    printf '%s  %s\n' "$3" "$2" | sha256sum -c -
}
if command -v uv >/dev/null && [[ "$(uv --version)" == 'uv 0.11.23 '* ]]; then
    uv_command="$(command -v uv)"
else
    # Bootstrap from the verified PyPI wheel inside the checkout.
    for tool in curl sha256sum unzip; do
        command -v "$tool" >/dev/null || { echo "Install $tool before running this script." >&2; exit 2; }
    done
    wheel_host=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages
    if $upstream; then wheel_host=https://files.pythonhosted.org/packages; fi
    printf 'Downloading pinned uv wheel from %s\n' "$wheel_host"
    download "$wheel_host/$wheel" "$bootstrap/uv.whl" "$hash"
    unzip -p "$bootstrap/uv.whl" "uv-0.11.23.data/scripts/$uv_name" > "$bootstrap/$uv_name"
    chmod +x "$bootstrap/$uv_name"
    uv_command="$bootstrap/$uv_name"
fi

export PYTHONUTF8=1
unset UV_INDEX UV_DEFAULT_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_FIND_LINKS
flags=()
if $upstream; then flags+=(--upstream); fi

python_request=3.12
if ! "$uv_command" --no-config python find --no-python-downloads "$python_request" >/dev/null 2>&1; then
    python_request="$(pwd)/$python_bin"
    if [[ ! -f "$python_request" ]]; then
        for tool in curl tar sha256sum; do
            command -v "$tool" >/dev/null || { echo "Install $tool or Python 3.12 and retry." >&2; exit 2; }
        done
        python_host=https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone
        if $upstream; then
            python_host=https://github.com/astral-sh/python-build-standalone/releases/download
        fi
        printf 'Downloading pinned Python 3.12.16 build from %s\n' "$python_host"
        download "$python_host/20261026/cpython-3.12.16%2B20261026-$python_platform-install_only_stripped.tar.gz" \
            "$bootstrap/python.tar.gz" "$python_hash"
        mkdir -p -- .venv-course-python
        tar -xzf "$bootstrap/python.tar.gz" -C .venv-course-python
    fi
fi
"$uv_command" --no-config run --no-project --no-python-downloads --python "$python_request" python \
    tools/install_course.py --backend "$backend" --uv "$uv_command" "${flags[@]}"
