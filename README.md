# Off-Axis Freeform Illumination Lens Design via Optimal Transport

**Keywords:** optimal transport · caustic · illumination lens design · off-axis ·
freeform optics · differentiable ray tracing · inverse rendering

## Overview

This repository contains research code for designing a single freeform
refractive surface that reshapes a collimated beam into a prescribed irradiance
pattern under large off-axis deflection. In computer graphics this is a
**caustic** design problem: the freeform surface is the refracting element that
produces a desired light pattern. In illumination optics it is an inverse
freeform lens design problem.

The pipeline combines:

1. a reflection-aware semi-discrete optimal transport initialization;
2. B-spline surface fitting with triangulated correspondence refinement;
3. end-to-end differentiable ray tracing with Fresnel losses.

The technical notes are in [`method.md`](method.md). This is research code,
not a production optical design tool.

The manuscript was not accepted for publication. Rather than leave the work
unused, the code is released here in the hope that it saves someone else time
on the same problem.

## Repository layout

- `spline_irra.py` — main optimization driver.
- `spline_rl_render.py`, `mesh_output.py` — checkpoint rendering and mesh export.
- `utils/`, `models/` — spline model, losses, sampling, and optimization code.
- `Optimal_Transport_rl/` — CUDA/C++ semi-discrete optimal transport solver.
- `render_exe_program/` — pybind11 ray-tracing extension.
- `mitsuba_render/` — optional Mitsuba validation scripts.
- `configs/` — algorithm parameters.
- `data/` — sample input data.
- `method.md` — method notes.

Generated output (`log/`, `result_sp_data/`, `result_spline/`) is not tracked.

## 使用说明

### 环境要求

- Linux + NVIDIA GPU（已安装驱动）
- 已安装 conda（Miniconda / Anaconda）
- 系统 CUDA toolkit：需要 `nvcc`、cublas 和 curand 开发头文件
- 磁盘预留约 10 GB

### 1. 配置环境

```bash
bash setup_caustic_env.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caustic_env
```

默认环境名为 `caustic_env`，可通过 `ENV_NAME=xxx bash setup_caustic_env.sh` 修改。

### 2. 编译最优传输程序

```bash
cd Optimal_Transport_rl
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cd ..
```

CUDA 默认自动探测；如需指定：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_ROOT=/usr/local/cuda \
      -DCAUSTIC_CUDA_ARCH=86
```

### 3. 安装 pybind11 扩展

```bash
cd render_exe_program
python setup.py install
cd ..
```

如需固定 GPU 架构，可在安装前设置 `TORCH_CUDA_ARCH_LIST`，例如
`export TORCH_CUDA_ARCH_LIST="8.6"`。

### 4. 运行

```bash
bash run_spline_irra.sh
```

也可以直接运行：

```bash
python spline_irra.py
```

运行日志写入 `log/`，中间数据写入 `result_sp_data/`，最终结果写入
`result_spline/`。

### 验证渲染（可选）

验证代码包括网格提取 `mesh_output.py`、渲染模拟
`mitsuba_render/render_off.py`（未考虑 gamma 矫正）、差异计算
`mitsuba_render/pic_diff.py`，需要 `mitsuba`、`scikit-image` 等依赖。建议单独
创建验证环境，不要与 `caustic_env` 合并：

```bash
conda create -n mitsuba3_env -c conda-forge python=3.10 pip -y
conda activate mitsuba3_env
pip install mitsuba scikit-image pillow
```

确认可用的渲染变体：

```bash
python -c "import mitsuba as mi; print(mi.variants())"
```

### 参数设置

全过程参数保存在 `configs/_py_sp_settings.json`。参数维度较多，建议结合代码
阅读。

## License

MIT License. See [`LICENSE`](LICENSE).
