#!/usr/bin/env bash
# Keeps every cache inside the project folder and sets GPU-safe defaults.
export PROJECT_ROOT="/mnt/c/Users/Griff/Projects/peg-insertion"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.cache/uv-python"
export UV_PYTHON_PREFERENCE=only-system
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/mpl"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_ROOT/.cache/jax"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export WARP_CACHE_PATH="$PROJECT_ROOT/.cache/warp"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
# (XLA_FLAGS is set inside scripts/train.py, not here, to avoid duplicating the flag)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
source "$PROJECT_ROOT/.venv/bin/activate" 2>/dev/null || true
