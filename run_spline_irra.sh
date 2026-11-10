#!/bin/bash
# 运行 spline_irra.py
#   - 程序全部输出(Python + C++ 子进程 + irra_render 扩展)写入 log/<时间戳>.txt
#   - 终端保持干净,只在结束时报一行日志路径
# 用法:  bash run_spline_irra.sh
# 后台跑: nohup bash run_spline_irra.sh >/dev/null 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-caustic_env}"   # spline_irra.py 需要 torch / irra_render,不是 mitsuba3_env
PROJ_DIR="${PROJ_DIR:-$SCRIPT_DIR}"
SCRIPT="$PROJ_DIR/spline_irra.py"
LOG_DIR="$PROJ_DIR/log"
CPP_BIN="$PROJ_DIR/Optimal_Transport_rl/build/caustic_GPU"

# 自动定位 conda（可用 CONDA_ROOT 覆盖）
if [ -z "${CONDA_ROOT:-}" ]; then
    CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
    if [ -z "$CONDA_BIN" ]; then
        for _c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
                  "$HOME/miniforge3/bin/conda" /opt/conda/bin/conda \
                  /usr/local/miniconda3/bin/conda; do
            [ -x "$_c" ] && { CONDA_BIN="$_c"; break; }
        done
    fi
    if [ -n "$CONDA_BIN" ]; then
        CONDA_ROOT="$("$CONDA_BIN" info --base 2>/dev/null || true)"
    fi
fi
if [ -z "${CONDA_ROOT:-}" ] || [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "[spline_irra] 找不到 conda，请设置 CONDA_ROOT 或确保 conda 在 PATH 中" >&2
    exit 1
fi

# ============ 日志文件 ============
mkdir -p "$LOG_DIR" || exit 1
LOG_FILE="$LOG_DIR/spline_irra_$(date +%Y%m%d_%H%M%S).txt"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest.txt"

# 保存终端 stdout,只用于最后报日志路径;之后所有输出都进 log
exec 9>&1

SECONDS=0
trap 'echo "[$(date "+%F %T")] !! 收到中断信号,进程被终止" >> "$LOG_FILE"; exit 130' INT TERM

# ============ 环境 ============
{
  echo "============================================================"
  echo "start    : $(date '+%F %T')"
  echo "host     : $(hostname)"
  echo "script   : $SCRIPT"
  echo "env      : $ENV_NAME"
  echo "cwd      : $PROJ_DIR"
  echo "============================================================"
} >> "$LOG_FILE"

source "$CONDA_ROOT/etc/profile.d/conda.sh" >> "$LOG_FILE" 2>&1
if ! conda activate "$ENV_NAME" >> "$LOG_FILE" 2>&1; then
    echo "conda activate $ENV_NAME 失败" >> "$LOG_FILE"
    echo "[spline_irra] 环境激活失败,详见: $LOG_FILE" >&9
    exit 1
fi

cd "$PROJ_DIR" || { echo "cd $PROJ_DIR 失败" >> "$LOG_FILE"; exit 1; }

# ============ 预检 ============
{
  echo "python   : $(command -v python)  ($(python --version 2>&1))"
  echo "---------------- 依赖检查 ----------------"
} >> "$LOG_FILE"

python - <<'PY' >> "$LOG_FILE" 2>&1
import importlib
bad = []
for m in ("torch", "numpy", "PIL", "scipy", "irra_render"):
    try:
        mod = importlib.import_module(m)
        print(f"  [ok]   {m:12s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        bad.append(m)
        print(f"  [FAIL] {m:12s} {type(e).__name__}: {e}")
if bad:
    print(f"\n缺少依赖: {', '.join(bad)}")
    raise SystemExit(1)
PY

if [ $? -ne 0 ]; then
    echo "依赖不满足,中止。详见: $LOG_FILE" >> "$LOG_FILE"
    echo "[spline_irra] 依赖缺失,中止。日志: $LOG_FILE" >&9
    exit 2
fi

[ -x "$CPP_BIN" ] || echo "[warn] 找不到 C++ 可执行文件: $CPP_BIN" >> "$LOG_FILE"

{
  echo "---------------- 开始运行 ----------------"
} >> "$LOG_FILE"

# ============ 运行 ============
# python -u      : Python 端 stdout/stderr 不缓冲,print 立即落盘
# stdbuf -o0 -e0 : 通过 LD_PRELOAD 让 C++ 子进程(caustic_GPU)的 stdio 也不缓冲
#                  LD_PRELOAD 会沿 fork/exec 继承,所以孙进程同样生效
stdbuf -o0 -e0 python -u "$SCRIPT" >> "$LOG_FILE" 2>&1
RC=$?

# ============ 收尾 ============
{
  echo "---------------- 结束 ----------------"
  echo "exit code: $RC"
  echo "end      : $(date '+%F %T')"
  echo "elapsed  : ${SECONDS}s"
  echo "============================================================"
} >> "$LOG_FILE"

# 唯一一行终端输出;想连这行都不要就注释掉下面这句
echo "[spline_irra] 完成 (exit=$RC),输出见: $LOG_FILE" >&9

exit $RC
