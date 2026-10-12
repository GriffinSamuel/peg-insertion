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

## Training throughput (final experiment)

All runs: Warp backend, 1024 environments, 20M environment steps, one at a time on the GPU
(the reward-check runs in results/archive/ ran four at once and are not listed). Wall time
includes the initial JIT/eval (~20 s with a warm compilation cache). Env steps/s is measured
from the first to the last in-training evaluation.

| run | wall time (min) | env steps/s | success (500 ep) |
|---|---|---|---|
| pos_2mm_s0 | 4.8 | 73,991 | 99.6% |
| pos_2mm_s1 | 5.7 | 65,400 | 97.6% |
| pos_2mm_s2 | 5.5 | 67,753 | 100.0% |
| pos_2mm_s3 | 5.7 | 65,132 | 97.0% |
| pos_2mm_s4 | 5.5 | 67,495 | 100.0% |
| pos_1mm_s0 | 4.9 | 72,743 | 96.2% |
| pos_1mm_s1 | 5.6 | 65,496 | 95.8% |
| pos_1mm_s2 | 5.4 | 69,002 | 0.0% |
| pos_1mm_s3 | 5.5 | 66,810 | 48.2% |
| pos_1mm_s4 | 5.5 | 67,316 | 0.0% |
| pos_0.5mm_s0 | 4.8 | 74,886 | 0.0% |
| pos_0.5mm_s1 | 5.6 | 66,178 | 78.6% |
| pos_0.5mm_s2 | 5.3 | 70,504 | 0.0% |
| pos_0.5mm_s3 | 5.5 | 66,895 | 4.6% |
| pos_0.5mm_s4 | 5.6 | 66,726 | 1.6% |
| force_2mm_s0 | 5.7 | 65,337 | 99.0% |
| force_2mm_s1 | 5.6 | 65,941 | 94.6% |
| force_2mm_s2 | 5.8 | 63,412 | 100.0% |
| force_1mm_s0 | 5.6 | 65,635 | 99.2% |
| force_1mm_s1 | 5.5 | 67,291 | 0.4% |
| force_1mm_s2 | 5.7 | 65,418 | 99.4% |
| force_0.5mm_s0 | 4.8 | 75,349 | 98.2% |
| force_0.5mm_s1 | 5.3 | 70,428 | 0.0% |
| force_0.5mm_s2 | 5.5 | 66,603 | 62.8% |
| force_0.5mm_s3 | 5.6 | 65,929 | 97.6% |
| force_0.5mm_s4 | 5.6 | 65,825 | 85.2% |

Backend benchmark (0.5M steps, 1024 envs, early in Phase 4): JAX backend ~2.5k steps/s steady,
Warp backend ~11k steps/s on the same short benchmark; the long runs above reach ~68k steps/s
once the compilation cache is warm and the run is long enough to amortise the evaluations.

## uv pip freeze

```
absl-py==2.5.1
aiofiles==25.1.0
asttokens==3.0.2
blinker==1.9.0
brax==0.14.2
click==8.5.0
contourpy==1.4.0
cycler==0.12.1
etils==1.14.0
executing==2.3.0
flask==3.1.3
flask-cors==6.0.5
flax==0.12.10
fonttools==4.66.1
fsspec==2026.9.0
glfw==2.10.2
humanize==4.16.0
imageio==2.38.1
imageio-ffmpeg==0.6.0
iniconfig==2.3.1
ipython==9.17.1
ipython-pygments-lexers==1.1.1
itsdangerous==2.2.0
jax==0.11.2
jax-cuda12-pjrt==0.11.2
jax-cuda12-plugin==0.11.2
jaxlib==0.11.2
jaxopt==0.8.5
jedi==0.20.0
jinja2==3.1.6
kiwisolver==1.5.1
lxml==6.1.3
markdown-it-py==4.2.0
markupsafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
mdurl==0.1.2
mediapy==1.2.7
ml-collections==1.1.0
ml-dtypes==0.6.0
msgpack==1.2.3
mujoco==3.15.0
mujoco-mjx==3.15.0
mujoco-warp==3.15.0
numpy==2.5.4
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-cccl-cu12==12.9.27
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvcc-cu12==12.9.86
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.27.0.42
nvidia-cufft-cu12==11.4.1.4
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nccl-cu12==2.32.3
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu12==3.8.0
opt-einsum==3.4.0
optax==0.2.8
orbax-checkpoint==0.12.7
packaging==26.3
pandas==3.0.6
parso==0.8.7
pexpect==4.9.0
pillow==12.3.0
-e file:///mnt/c/Users/Griff/Projects/peg-insertion/third_party/mujoco_playground
pluggy==1.6.0
prometheus-client==0.26.0
prompt-toolkit==3.0.53
protobuf==7.36.2
psutil==7.2.2
ptyprocess==0.7.0
pure-eval==0.2.4
pygments==2.21.0
pyopengl==3.1.10
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
pyyaml==6.0.3
rich==15.0.0
scipy==1.18.1
simplejson==4.2.0
six==1.17.0
stack-data==0.6.3
tensorboardx==2.6.5
tensorstore==0.1.85
tqdm==4.70.1
traitlets==5.16.1
treescope==0.1.10
trimesh==5.1.1
typing-extensions==4.16.0
uvloop==0.23.0
warp-lang==1.18.0
wcwidth==0.9.2
websockets==17.2
werkzeug==3.1.9
zipp==4.1.1
```
