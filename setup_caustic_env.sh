#!/usr/bin/env bash
# =============================================================================
# caustic_env 配置脚本（尽量小磁盘占用）
#   * 使用系统 CUDA toolkit，不安装 conda 的 cuda-toolkit
#   * torch 2.4.1 + cu121（pip，--no-cache-dir）
#   * C++ 依赖 Eigen / Boost / CGAL 5.5 / nlohmann_json 走 conda-forge
#   * 结束自动清理 conda 包缓存与 pip 缓存
#
# 用法：
#   bash setup_caustic_env.sh
#
# 说明：
#   - conda 位置自动探测（CONDA_EXE，或 PATH / 常见安装目录）
#   - 环境名默认 caustic_env，可用 ENV_NAME=xxx bash setup_caustic_env.sh 覆盖
#   - 激活时自动 export CUDA_HOME
#   - 不会自动编译 Optimal_Transport_rl / irra_render，编译命令在脚本末尾打印
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-caustic_env}"
PY_VER="${PY_VER:-3.10}"
TORCH_VER="${TORCH_VER:-2.4.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# 自动定位 conda：CONDA_EXE → PATH → 常见安装目录
CONDA="${CONDA_EXE:-}"
if [ -z "$CONDA" ] || [ ! -x "$CONDA" ]; then
    CONDA="$(command -v conda || true)"
fi
if [ -z "$CONDA" ] || [ ! -x "$CONDA" ]; then
    for _c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
              "$HOME/miniforge3/bin/conda" /opt/conda/bin/conda \
              /usr/local/miniconda3/bin/conda; do
        [ -x "$_c" ] && { CONDA="$_c"; break; }
    done
fi
[ -n "$CONDA" ] && [ -x "$CONDA" ] \
    || die "找不到 conda。请先安装 Miniconda/Anaconda，或设置 CONDA_EXE=/path/to/conda"

CONDA_ROOT="$("$CONDA" info --base 2>/dev/null || true)"
[ -n "$CONDA_ROOT" ] || die "无法获取 conda 根目录（conda info --base 失败）"
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"

# -----------------------------------------------------------------------------
# 1. 定位系统 CUDA toolkit（优先 PATH 中的 nvcc）
# -----------------------------------------------------------------------------
NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        NVCC="$(command -v nvcc)"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        NVCC="/usr/local/cuda/bin/nvcc"
    elif [ -x /usr/local/cuda-12.1/bin/nvcc ]; then
        NVCC="/usr/local/cuda-12.1/bin/nvcc"
    fi
fi
[ -n "$NVCC" ] || die "找不到 nvcc。请确认系统 CUDA toolkit（如 /usr/local/cuda-12.1）存在"

CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd)"
NVCC_REL="$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')"
log "CUDA toolkit : ${CUDA_HOME} (nvcc ${NVCC_REL:-unknown})"

[ -f "${CUDA_HOME}/targets/x86_64-linux/include/cublas_v2.h" ] \
    || die "CUDA toolkit 缺少 cublas 开发头文件: ${CUDA_HOME}/targets/x86_64-linux/include"
[ -f "${CUDA_HOME}/targets/x86_64-linux/include/curand.h" ] \
    || die "CUDA toolkit 缺少 curand 开发头文件: ${CUDA_HOME}/targets/x86_64-linux/include"

# CUDA 主版本需与 torch wheel 的主版本一致（12.x 之间适用 minor version compatibility）
case "$NVCC_REL" in
    12.*) : ;;
    *)    warn "系统 CUDA 为 ${NVCC_REL}，torch wheel 为 cu121，可能存在兼容性问题" ;;
esac

# -----------------------------------------------------------------------------
# 2. 创建环境
# -----------------------------------------------------------------------------
if [ -d "$ENV_PREFIX" ]; then
    log "环境已存在，跳过创建: $ENV_PREFIX"
else
    log "创建 conda 环境 ${ENV_NAME} (python ${PY_VER})"
    "$CONDA" create -y -n "$ENV_NAME" -c conda-forge --override-channels \
        --no-default-packages "python=${PY_VER}" pip
fi

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

# -----------------------------------------------------------------------------
# 3. conda-forge 依赖（C++ + Python）
#    不装 boost 元包：cgal 会带上 boost-cpp；缺 CMake config 时再补
# -----------------------------------------------------------------------------
NEED_CONDA_CMAKE=0
if command -v cmake >/dev/null 2>&1; then
    SYS_CMAKE_VER="$(cmake --version | awk 'NR==1{print $3}')"
    if [ "$(printf '%s\n%s\n' '3.20' "$SYS_CMAKE_VER" | sort -V | head -n1)" != '3.20' ]; then
        NEED_CONDA_CMAKE=1
    else
        log "使用系统 cmake ${SYS_CMAKE_VER}"
    fi
else
    NEED_CONDA_CMAKE=1
fi

CONDA_PKGS=(eigen nlohmann_json "cgal=5.5" numpy scipy pillow pybind11 ninja)
[ "$NEED_CONDA_CMAKE" -eq 1 ] && CONDA_PKGS+=(cmake)

log "安装 conda-forge 依赖: ${CONDA_PKGS[*]}"
"$CONDA" install -y -n "$ENV_NAME" -c conda-forge --override-channels "${CONDA_PKGS[@]}"

# Boost：CMakeLists 使用 module 模式，只需要头文件与库目录
if [ ! -f "$ENV_PREFIX/include/boost/version.hpp" ]; then
    warn "未检测到 Boost 头文件，补装 libboost-devel"
    if ! "$CONDA" install -y -n "$ENV_NAME" -c conda-forge --override-channels libboost-devel; then
        warn "libboost-devel 安装失败，回退 boost-cpp"
        "$CONDA" install -y -n "$ENV_NAME" -c conda-forge --override-channels boost-cpp
    fi
fi

# -----------------------------------------------------------------------------
# 4. PyTorch 2.4.1 (cu121)，不保留 pip 缓存
# -----------------------------------------------------------------------------
log "安装 torch ${TORCH_VER}+cu121（--no-cache-dir）"
"$PIP" install --no-cache-dir --index-url "$TORCH_INDEX" "torch==${TORCH_VER}"

# -----------------------------------------------------------------------------
# 5. 激活钩子：CUDA_HOME / PATH
# -----------------------------------------------------------------------------
mkdir -p "$ENV_PREFIX/etc/conda/activate.d" "$ENV_PREFIX/etc/conda/deactivate.d"
cat > "$ENV_PREFIX/etc/conda/activate.d/caustic_cuda.sh" <<EOF
# generated by setup_caustic_env.sh
export _CAUSTIC_OLD_CUDA_HOME="\${CUDA_HOME-}"
export CUDA_HOME="${CUDA_HOME}"
export PATH="\${CUDA_HOME}/bin\${PATH:+:\$PATH}"
EOF
cat > "$ENV_PREFIX/etc/conda/deactivate.d/caustic_cuda.sh" <<'EOF'
# generated by setup_caustic_env.sh
_caustic_cuda_bin="${CUDA_HOME:-}/bin"
export CUDA_HOME="${_CAUSTIC_OLD_CUDA_HOME-}"
unset _CAUSTIC_OLD_CUDA_HOME
if [ -n "${_caustic_cuda_bin}" ]; then
    export PATH="${PATH#${_caustic_cuda_bin}:}"
fi
unset _caustic_cuda_bin
EOF

# -----------------------------------------------------------------------------
# 6. 清理安装缓存，回收磁盘
# -----------------------------------------------------------------------------
log "清理 conda / pip 缓存"
"$CONDA" clean -afy >/dev/null 2>&1 || true
rm -rf "${HOME}/.cache/pip" 2>/dev/null || true

# -----------------------------------------------------------------------------
# 7. 验证
# -----------------------------------------------------------------------------
log "验证环境"
"$PY" - <<'PY'
import sys
print("python :", sys.version.split()[0])

import torch
print("torch  :", torch.__version__, "| torch CUDA:", torch.version.cuda,
      "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("gpu    :", torch.cuda.get_device_name(0), "| sm_%d%d" % cap)

import numpy, scipy, PIL
print("numpy  :", numpy.__version__)
print("scipy  :", scipy.__version__)
print("pillow :", PIL.__version__)

import pybind11
print("pybind11:", pybind11.__version__)
PY

log "依赖文件检查"
[ -d "$ENV_PREFIX/include/eigen3/Eigen" ]                && echo "  [ok] Eigen   $ENV_PREFIX/include/eigen3"
[ -d "$ENV_PREFIX/include/boost" ]                       && echo "  [ok] Boost   $ENV_PREFIX/include/boost"
compgen -G "$ENV_PREFIX/lib/cmake/CGAL/CGALConfig.cmake" >/dev/null && echo "  [ok] CGAL    $ENV_PREFIX/lib/cmake/CGAL"
[ -f "$ENV_PREFIX/share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake" ] \
    && echo "  [ok] nlohmann_json"
[ -x "$ENV_PREFIX/bin/ninja" ]                           && echo "  [ok] ninja"
[ -x "$CUDA_HOME/bin/nvcc" ]                             && echo "  [ok] nvcc    $CUDA_HOME/bin/nvcc"

# -----------------------------------------------------------------------------
# 8. 磁盘占用
# -----------------------------------------------------------------------------
echo
log "磁盘占用"
du -sh "$ENV_PREFIX" 2>/dev/null || true
df -h / 2>/dev/null || true

echo
cat <<EOF
============================================================
环境配置完成：$ENV_PREFIX

后续编译（脚本未自动执行）：

  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate ${ENV_NAME}

  # 1) 最优传输 C++ 程序
  cd "${SCRIPT_DIR}/Optimal_Transport_rl"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"\$(nproc)"

  # 2) pybind11 渲染扩展（安装进环境 site-packages，便于直接 import）
  cd "${SCRIPT_DIR}/render_exe_program"
  python -m pip install --no-build-isolation --no-cache-dir .

注意：
  * 运行前确认 data/pic_data/ 下的输入图片存在；
    result_sp_data/ 下的初值数据（data_point/2/）也需要存在。
  * 运行输出 result_sp_data/ 与 result_spline/ 体积较大，
    建议软链到大容量磁盘。
============================================================
EOF
