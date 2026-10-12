# Environment

Recorded 2026-10-12 (Phase 1).

## Hardware / OS
- Laptop: Intel Core Ultra 9 285H, 32 GB RAM (WSL2 sees 19 GiB + 8 GiB swap, 12 CPUs)
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, Blackwell (sm_120)
- Windows driver 596.58 (CUDA 13.2), WSL2 Ubuntu 24.04, Python 3.12.3

## Software
- jax 0.11.2 | mujoco/mjx 3.15.0 | brax 0.14.2 | flax 0.12.10 | warp-lang 1.18.0
- MuJoCo Playground commit d59156e (cloned into third_party/, installed editable)
- MuJoCo Menagerie commit 1b86ece (pinned by Playground)
- JAX backend: gpu, devices: [CudaDevice(id=0)]

## Timings
- `import jax, brax, mujoco_playground`: 36-60 s (project lives on /mnt/c, slow for many small files; acceptable band)
- Playground smoke test (PandaPickCubeCartesian, impl=jax, single env): first step incl. JIT compile 20.9 s; 100 single-env steps 7.96 s (Python-loop dispatch bound)
- osmesa render test: (240, 320, 3), mean 9.7 -> rendering works

## nvidia-smi
```
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 5070 Laptop GPU, 596.58, 8151 MiB
```
