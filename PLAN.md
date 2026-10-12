# PLAN.md — Learning tight-tolerance peg insertion with MuJoCo Playground on an RTX 5070

This is the step-by-step plan Claude Code follows. Each phase has: goal, exact steps, **Done when** checks, time budget, and **If it fails** instructions. Phases are sequential. Do not skip the smoke tests.

Research question: **How does RL insertion success fall as hole clearance tightens from 2 mm to 0.5 mm, and does wrist force sensing recover success at 0.5 mm?**

Total budget: 7 days of Claude Code work after Griffin's Day 0 setup. Suggested split: Phase 1 (0.5 d), Phase 2 (1 d), Phase 3 (0.5 d), Phase 4 (1 d), Phase 5 (0.5 d), Phase 6 (2 d), Phase 7 (1 d), Phase 8 (0.5 d).

---

## Day 0 — Griffin's one-time setup (human, Windows + WSL)

Claude Code cannot do these. Griffin does them once, in order. Each line is copy-paste.

### 0.1 Update the NVIDIA driver (Windows)
Install the latest GeForce Game Ready or Studio driver from nvidia.com (or GeForce Experience / NVIDIA App). Target **driver version 580 or newer** (needed for the CUDA 13 wheels that `warp-lang` ships; CUDA 12 JAX wheels need ≥ 525). Reboot.

### 0.2 Install WSL2 with Ubuntu 24.04 (PowerShell as Administrator)
```powershell
wsl --install -d Ubuntu-24.04
```
Reboot if asked. Open "Ubuntu 24.04" from the Start menu, create a Linux username and password when prompted. Then confirm WSL 2:
```powershell
wsl -l -v
```
(The VERSION column must say 2.)

### 0.3 Give WSL more memory (Windows)
Create the file `C:\Users\Griff\.wslconfig` with:
```ini
[wsl2]
memory=20GB
processors=12
swap=8GB
```
Then in PowerShell:
```powershell
wsl --shutdown
```
Reopen Ubuntu.

### 0.4 Check the GPU is visible inside WSL (Ubuntu terminal)
```bash
nvidia-smi
```
You should see "NVIDIA GeForce RTX 5070 Laptop GPU" and the driver version. If this fails, the Windows driver is too old or WSL needs updating: in PowerShell run `wsl --update`, reboot, retry. **Do not install any Linux NVIDIA driver inside WSL.**

### 0.5 Install system packages (Ubuntu terminal, needs your password)
```bash
sudo apt update && sudo apt install -y git curl jq ffmpeg build-essential python3.12 python3.12-venv libosmesa6 libosmesa6-dev libgl1 libglu1-mesa libglew-dev patchelf
```

### 0.6 Install `uv` (Python package manager) and the GitHub CLI
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc && uv --version
```
```bash
(type -p wget >/dev/null || sudo apt install -y wget) && sudo mkdir -p -m 755 /etc/apt/keyrings && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && sudo apt update && sudo apt install -y gh
```
Log in to GitHub (choose GitHub.com, HTTPS, answer **Y** to "Authenticate Git with your GitHub credentials?", then "Login with a web browser" and follow the code). Without the Y, the final `git push` would hang on a password prompt.
```bash
gh auth login
```
```bash
gh auth setup-git && gh auth status
```
Set your git identity (use your GitHub email):
```bash
git config --global user.name "Griffin" && git config --global user.email "griffinsamuel869@gmail.com"
```

### 0.7 Install Claude Code inside WSL and put the project's Claude settings in place
```bash
curl -fsSL https://claude.ai/install.sh | bash
```
The settings, hooks and subagent definitions were delivered in `claude-config/` (the `.claude` folder cannot be written remotely). Move them into place and make the hooks executable:
```bash
cd /mnt/c/Users/Griff/Projects/peg-insertion && mkdir -p .claude && cp -r claude-config/. .claude/ && chmod +x .claude/hooks/*.sh && rm -rf claude-config && ls -R .claude
```
You should see `settings.json`, `hooks/block-outside-project.sh`, `hooks/guard-bash.sh`, `agents/fable-debugger.md`, `agents/opus-debugger.md`.
Then open the project folder and start Claude Code:
```bash
cd /mnt/c/Users/Griff/Projects/peg-insertion && claude
```
Accept the workspace trust prompt (project settings only apply after that). Log in when prompted. In the Claude Code prompt, type `/model` and confirm two things: the main model is **Opus** (pick it if not), and **Fable** appears in the list. If Fable is not listed, add the sentence noted in `START_PROMPT.md` to your first message.

**Trigger the Fable billing consent now, not mid-run.** On some plans the first Fable use shows a consent prompt about usage credits; if that happens unattended at the first escalation, the run idles until you check in. So type this as a message once, and answer any prompt:
```
Use the fable-debugger subagent to reply with the single word OK, then stop.
```

### 0.8 Keep the laptop awake and un-rebooted
Plug in. Windows Settings → System → Power → "Screen and sleep" → Never while plugged in. Windows Settings → Windows Update → **Pause updates for 1 week** (an update reboot kills the run). Control Panel → Power Options → "Choose what the power buttons do" → untick "Turn on fast startup".

### 0.9 Kick off, then check in twice a day
Paste the contents of `START_PROMPT.md` as your first message. Claude is instructed to keep working until `DONE.md` or `BLOCKED.md` exists, but three things can still idle the session: Claude ending a turn between phases, **your plan's usage limit** (Claude Code shows a message and waits for the window to reset), and a WSL/laptop hiccup. So glance at the terminal **morning and evening**. If it is idle with no `DONE.md`, paste the contents of `RESUME_PROMPT.md` (one paste restarts it from where it left off; nothing is lost because every phase is committed and logged). If `BLOCKED.md` exists, read it; it says what you need to do. `PROGRESS.md` always shows where things stand.

---

## Phase 1 — Environment setup inside the project (Claude Code)

**Goal:** a working Python environment, inside the project folder, with JAX seeing the GPU and MuJoCo Playground importing.

### 1.1 Create the layout
```
peg-insertion/
  .claude/           (settings, hooks, agents — already present)
  CLAUDE.md PLAN.md START_PROMPT.md PROGRESS.md DECISIONS.md
  .gitignore .python-version pyproject.toml
  scripts/env.sh scripts/train.sh scripts/eval.sh scripts/sweep.sh
  scripts/train.py scripts/evaluate.py scripts/plot.py scripts/render.py scripts/scripted_insert.py
  peg_insertion/__init__.py peg_insertion/env.py peg_insertion/configs.py peg_insertion/xml_builder.py
  peg_insertion/assets/   (generated XMLs)
  tests/test_env.py tests/test_scripted.py
  logs/  checkpoints/  results/videos/  .cache/  .venv/   (all gitignored except results/ summaries)
  third_party/mujoco_playground/   (git clone, gitignored)
```
Initialize git now: `git init -b main`, `git config core.autocrlf false` (the folder is on a Windows drive), the provided `.gitignore` already excludes venv, caches, logs, checkpoints and `third_party/`; first commit.

### 1.2 Write `scripts/env.sh` (sourced by every script)
```bash
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
```

### 1.3 Clone Playground into the project and create the venv
```bash
cd /mnt/c/Users/Griff/Projects/peg-insertion && source scripts/env.sh
git clone --depth 1 https://github.com/google-deepmind/mujoco_playground.git third_party/mujoco_playground
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 1.4 Install everything in ONE resolve (so the GPU plugin and `jax` versions stay matched), in the background
Installing Playground after JAX would re-resolve `jax` and can silently leave a mismatched `jax-cuda12-plugin` → CPU fallback. The install downloads several GB of CUDA wheels onto a slow drive, so it must not run in a foreground call (a timeout mid-install leaves a half-populated venv):
```bash
mkdir -p logs && setsid nohup uv pip install -e third_party/mujoco_playground "jax[cuda12]" tensorboardX PyOpenGL mediapy imageio imageio-ffmpeg pytest matplotlib pandas pyyaml pillow > logs/install.log 2>&1 < /dev/null &
```
Poll with `sleep 120 && tail -n 5 logs/install.log` until the final "Installed N packages" line. Then:
```bash
python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
python -c "import mujoco_playground, mujoco, mujoco.mjx as mjx, brax, tensorboardX; print(mujoco.__version__, brax.__version__)"
```
Expected: JAX ≥ 0.6, backend `gpu`, one `CudaDevice`. Do **not** use the `[all]` extra (pulls `rsl-rl-lib` → torch ≈2.5 GB; we only use the Brax PPO path). `tensorboardX` is required or `--use_tb` silently writes nothing.

**If backend is `cpu`:** (1) `nvidia-smi` works? (2) `echo $LD_LIBRARY_PATH` must be empty; (3) `uv pip list | grep -i -E "jax|nvidia"` — `jax`, `jaxlib`, `jax-cuda12-plugin`, `jax-cuda12-pjrt` must share one version; if not, `uv pip install -U "jax[cuda12]"` and recheck; (4) try `uv pip install -U "jax[cuda13]"` (needs Windows driver ≥ 580). Still CPU → **T2**.

### 1.4b Fetch the menagerie assets in the background (large clone; never do it in a foreground call)
First `registry.load` clones the full mujoco_menagerie history (hundreds of MB). If a foreground call times out mid-clone, a half-populated directory is left behind and `ensure_menagerie_exists()` never retries. So:
```bash
mkdir -p logs && setsid nohup python -c "from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists(); print('MENAGERIE OK')" > logs/menagerie.log 2>&1 < /dev/null &
```
Poll with `sleep 60 && tail -n 5 logs/menagerie.log` until `MENAGERIE OK`. **If it fails** or `third_party/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/` exists but is incomplete: `rm -rf` that one directory (it is inside the project) and rerun.

### 1.5 First GPU smoke run (Playground's own env) — proves the whole stack
Write this to `scripts/smoke_playground.py` and run it in the background (first sm_120 JIT compile may take 5–15 minutes):
```python
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import time, jax, jax.numpy as jp
from mujoco_playground import registry
# Force the JAX backend: the env's default_config has impl='warp'.
env = registry.load("PandaPickCubeCartesian", config_overrides={"impl": "jax"})
reset, step = jax.jit(env.reset), jax.jit(env.step)
t = time.time(); s = reset(jax.random.PRNGKey(0)); s = step(s, jp.zeros(env.action_size))
jax.block_until_ready(s.obs); print("first step incl. compile (s):", round(time.time() - t, 1))
t = time.time()
for _ in range(100): s = step(s, jp.zeros(env.action_size))
jax.block_until_ready(s.obs); print("100 steps (s):", round(time.time() - t, 3))
print("SMOKE OK")
```
```bash
setsid nohup python scripts/smoke_playground.py > logs/smoke_playground.log 2>&1 < /dev/null &
```
Poll with `sleep 120 && tail -n 5 logs/smoke_playground.log`. (`obs` is a flat array for non-vision envs, hence `block_until_ready(s.obs)`, not `s.obs['state']`.) **Done when** the log ends with `SMOKE OK`.

### 1.5b Import-speed check (the project lives on the Windows drive, which is slow for many small files)
```bash
time python -c "import jax, brax, mujoco_playground"
```
Record the time in `results/environment.md`. Under 30 s: fine. 30–90 s: acceptable, but batch probes into scripts rather than many `python -c` calls. Over 90 s: note it in `PROGRESS.md` as a time-budget risk and prefer fewer, longer runs; do not relocate anything outside the project folder.

### 1.6 Rendering smoke test
```bash
python -c "
import os; os.environ['MUJOCO_GL']='osmesa'
import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>')
d = mujoco.MjData(m); r = mujoco.Renderer(m, 240, 320); mujoco.mj_forward(m, d); r.update_scene(d); img = r.render(); print(img.shape, img.mean())"
```
**Done when** it prints `(240, 320, 3)` and a mean > 0. If osmesa fails, try `MUJOCO_GL=egl` once; if both fail, continue the project and mark videos as a known gap in `PROGRESS.md` (do not stop for this).

### 1.7 Record environment facts
Save `python -c "import jax, mujoco, brax, mujoco_playground; ..."` versions, `nvidia-smi` output, and the timings from 1.5 into `results/environment.md`. Commit: "Phase 1: environment ready".

Also write the deadline at the top of `PROGRESS.md`: `Deadline: <today + 7 days>` so trigger T7 is checkable.

**Phase 1 done when:** 1.4 prints `gpu`, 1.4b and 1.5 end with their OK lines, 1.6 renders (or is logged as a gap), commit exists. → Append to `PROGRESS.md` and start Phase 2 immediately in this same turn.

---

## Phase 2 — Build the peg-insertion environment

**Goal:** a Playground-style `MjxEnv` named `PegInsertion` with a configurable clearance and an optional force observation, registered so the trainer can load it.

### 2.1 Scene design (write `peg_insertion/xml_builder.py`, which generates XML strings)
Start from Playground's `third_party/mujoco_playground/mujoco_playground/_src/manipulation/franka_emika_panda/xmls/mjx_scene.xml` (keep its `<option>` block: `timestep=0.005 iterations=5 ls_iterations=8 integrator=implicitfast` and `<flag eulerdamp="disable"/>`, but set `max_contact_points` to 20, see Contacts below), and its include of the menagerie `mjx_panda.xml`. Remove the cube. Add:

- **Attaching the peg to the hand.** The Panda arrives through `<include file="mjx_panda.xml"/>`, and that file comes from the menagerie assets (`panda.get_assets()`), not from Playground's `xmls/`. You cannot add a child to an included body from the parent file. So in `xml_builder.py`: `assets = panda.get_assets()`; take `assets['mjx_panda.xml']` (bytes → str), insert the peg body XML immediately before the line containing `<site name="gripper"` inside the `hand` body, and pass the patched assets dict to `mujoco.MjModel.from_xml_string(scene_xml, assets)`. (Alternative if string patching is fragile: `spec = mujoco.MjSpec.from_string(scene_xml, assets)`, `spec.body('hand').add_body(...)`, `spec.compile()`.)
- **Peg**: `<body name="peg" pos="0 0 {0.100 + pl/2 - 0.01}">` as a jointless child of `hand` (rigidly attached = the "already grasped" simplification). In the hand frame **+z points toward the fingertips**; the TCP (`gripper` site) is at z = 0.100 and the pads end near 0.112. So the peg base sits 1 cm inside the fingers and the tip protrudes `tip_offset = pl − 0.01` below the TCP. Geom: `<geom name="peg_geom" type="box" size="{pw/2} {pw/2} {pl/2}" mass="0.05" rgba="0.2 0.4 0.9 1" contype="0" conaffinity="0"/>` with `pl` = 0.06 m, `pw` = 0.020 m. Sites: `<site name="peg_tip" pos="0 0 {+pl/2}"/>`, `<site name="peg_base" pos="0 0 {-pl/2}"/>`, `<site name="ft_site" pos="0 0 {-pl/2}"/>`. `tests/test_env.py` must assert, at the keyframe, world `tip_z < hand_z` and `|tip − TCP| ≈ pl − 0.01`.
- **Force sensor**: MuJoCo's `<force>` sensor reports the interaction force between a child body and its parent, in the site frame, and "often requires creating a dummy body welded to its parent." The jointless peg body is exactly that. `<sensor><force name="peg_force" site="ft_site"/><torque name="peg_torque" site="ft_site"/><framepos name="tip_pos" objtype="site" objname="peg_tip"/><framelinvel name="tip_vel" objtype="site" objname="peg_tip"/><framepos name="hole_pos" objtype="site" objname="hole_entrance"/></sensor>`. Note the sensor reads the peg's own weight (≈0.5 N) in free space; the test asserts `|F| < 1 N` free, `> 2 N` when pressed into the table.
- **Table**: `<geom name="table" type="box" size="0.4 0.4 0.02" pos="0.5 0 -0.02" contype="0" conaffinity="0"/>` (top at z=0). **Remove the `floor` plane from the copied scene** (it is coplanar with the table top and would double the bottom contacts).
- **Hole block**: `<body name="hole_block" mocap="true" pos="0.5 0 0">` with 4 box walls (`wall_px, wall_nx, wall_py, wall_ny`) forming a square hole of inner width `hw = pw + 2*clearance`, wall thickness 0.02, height `hd` = 0.03 (hole depth), sitting on the table; hole floor = table. Walls at `±(hw/2 + 0.01)`, all `contype="0" conaffinity="0"`. Sites `hole_entrance` at `(0,0,hd)` and `hole_bottom` at `(0,0,0)`. Mocap lets `reset()` randomize the hole by writing `data.mocap_pos` (MJX `make_data` copies `mocap_pos` from `body_pos`; mocap bodies work in MJX kinematics).
- **Contacts — all explicit pairs, no masks.** Fingertip pads have `conaffinity=3`, so a default peg would permanently penetrate them and pollute the force sensor. With every new geom at `contype=0 conaffinity=0`, drive collisions only through `<contact><pair geom1="peg_geom" geom2="wall_px" condim="3" solref="0.01 1" friction="0.6 0.005 0.0001"/>` × 4 walls + `table` (pairs ignore contype masks; MJX supports pairs). Set `<custom><numeric name="max_contact_points" data="20"/>` (5 box-box pairs × up to 8 points; 12 can drop the bottom contact when the peg touches two walls). `naconmax = 20 * num_envs` in config, scaled with `num_envs`.
- **Keyframe**: the copied scene has none (Playground's live in `mjx_single_cube.xml` with the cube's qpos). Emit `<keyframe><key name="home" qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 0.01 0.01" ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.01"/></keyframe>` (gripper at 0.01 = half the peg width, so fingers visibly hold the peg instead of closing through it).

`build_xml(clearance_m: float, peg_width=0.020, peg_len=0.06, hole_depth=0.03) -> str` returns the full XML. Also write generated XMLs to `peg_insertion/assets/scene_c{clearance_mm}.xml` for inspection and tests.

### 2.2 Env class (write `peg_insertion/env.py`)
Model it on `pick_cartesian.py`. Key parts:

```python
def default_config():
  return config_dict.create(
    ctrl_dt=0.02, sim_dt=0.005, episode_length=200, action_repeat=1,
    action_scale=0.002,            # metres per step per axis (fine control for insertion)
    clearance_mm=2.0, use_force_obs=False,
    hole_xy_noise=0.01,            # metres: random hole offset per episode (uniform ±)
    obs_hole_bias_std=0.00075,     # metres: per-episode constant bias on observed hole position
                                   # (0.75 mm std: |bias| > 0.5 mm in ~50% of episodes per axis, so
                                   #  position-only control should fail often at 0.5 mm clearance
                                   #  while force feedback can still find the hole)
    obs_noise_std=0.0002,          # metres: per-step white noise on positions
    force_clip=20.0,               # N, for normalising force obs
    reward_config=config_dict.create(scales=config_dict.create(
        reach=1.0, align=1.0, insert=2.0, success=10.0, action_rate=-0.05, force_pen=-0.01, no_soln=-0.01)),
    impl='jax', naconmax=20*1024, njmax=128)   # naconmax = max_contact_points * num_envs; scale with num_envs
```
- `__init__`: build XML with `build_xml(clearance_mm/1000, ...)`, `mujoco.MjModel.from_xml_string(xml, assets=panda.get_assets())`, set `opt.timestep=sim_dt`, `mjx_env.put_model`. Cache ids for sites/sensors/actuators. Compute `_start_tip_transform` like Playground does. `action_size = 3` (dx, dy, dz).
- `reset(rng)`: sample hole xy offset within `±hole_xy_noise`, set `mocap_pos`; sample a per-episode observation bias (Normal, std `obs_hole_bias_std`); start the arm from the `home` keyframe then IK the tip to ~5 cm above the hole entrance with ±1 cm xy error. Store `current_pos`, `prev_action`, `bias`, `steps`, `succeeded`, `peak_force`, `first_success_step`, `rng` in `state.info`.
- **Per-episode randomization actually happening:** Playground's `wrap_for_brax_training` uses `BraxAutoResetWrapper(full_reset=False)` by default, which on `done` restores the cached first `data`/`obs` and never re-samples `info`. That would freeze hole position and bias for the whole run. **In `scripts/train.py` and `scripts/evaluate.py` pass `wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True)`** (the parameter exists; Brax calls `wrap_env_fn` with keyword args). Verify in `tests/test_env.py` that two consecutive auto-resets produce different `mocap_pos`.
- `MjxEnv` abstract members you must implement: `reset`, `step`, `xml_path`, `action_size`, `mj_model`, `mjx_model` (all six, or instantiation fails with TypeError).
- `step(state, action)`: `action = clip(action,-1,1)`; `tip_target = current_pos + action*action_scale`, clip to a workspace box around the hole: xy within ±0.05 m of the hole centre, **z ∈ [−(hole_depth + 0.005), 0.10] relative to the hole entrance** (the tip must be allowed to reach the bottom; success needs depth ≥ 0.8·hole_depth = 0.024 m). **Anti-windup clamp:** `tip_target_z = max(tip_target_z, measured_tip_z − 0.005)` so a blocked peg can never accumulate more than 5 mm of servo error (keeps contact forces in the 0–20 N band). Convert tip target to the TCP frame the IK uses: `tcp_target = tip_target + [0, 0, tip_offset]` with `tip_offset = pl − 0.01` (see 2.1). Fixed orientation (`_start_tip_transform[:3,:3]`); `panda_kinematics.compute_franka_ik(tcp_mat, q7_current, q_current)` → arm ctrl (fall back to previous ctrl and set `no_soln` if NaN); gripper ctrl fixed at `pw/2 = 0.01`. `mjx_env.step(model, data, ctrl, n_substeps)`. Compute obs, reward, done, metrics.
- `_get_obs(data, info)`: `rel = (tip_pos − hole_entrance_pos) + bias + noise` (3), `tip_vel` (3), `prev_action` (3), `insert_depth` (1), and if `use_force_obs`: `force/force_clip` clipped to [−1,1] (3) and `torque/clip` (3). Return `{'state': obs}`.
- `_get_reward(...)` using `reward_util.tolerance` from Playground (same as its own `AlohaSinglePegInsertion`):
  - `reach = tolerance(||rel_xyz||, bounds=(0,0.002), margin=0.08, sigmoid='linear')`
  - `align = tolerance(||rel_xy||, bounds=(0, clearance*0.5), margin=0.02)` gated by `tip above entrance` (z within 1 cm)
  - `insert = clip(depth / hole_depth, 0, 1)` where `depth = max(0, entrance_z − tip_z)`; only counts when `||rel_xy|| < clearance + pw/2` (i.e. actually in the hole)
  - `success_bonus`: paid **once**, on the first step where `depth ≥ 0.8*hole_depth`. Track `info['succeeded'] = max(prev, depth ≥ 0.8·hd)`; `newly = (depth ≥ 0.8·hd) & ~prev_succeeded`. **Do not terminate the episode on success** (terminating while paying dense reward every step teaches the policy to hover above the hole forever: dense return ≈ 0.29/(1−0.97) ≈ 9.7 vs a one-off bonus ≈ 0.7). Keep paying `insert` each step so the optimal policy stays inserted.
  - `action_rate = −||action − prev_action||²`, `force_pen = −clip(||force|| − 15, 0, 20)/20` (bounded; penalises crushing), `no_soln` when IK returns NaN.
  - Total = Σ scale×term. Normalise by Σ positive scales like Playground does.
- `done`: tip leaves workspace, or NaN in qpos. (Not success.)
- Metrics: Brax's EvalWrapper **sums** `state.metrics` over an episode, so only put per-step quantities there that make sense summed: `success = newly` (sums to 0/1 per episode), `insert_reward`, `reach_reward`, `force_pen`. Keep `depth`, `xy_err`, `peak_force`, `first_success_step`, `succeeded` in `state.info`; `evaluate.py` reads them from the final step of its own rollout.
- Register: in `peg_insertion/__init__.py`, `from mujoco_playground._src.manipulation import register_environment; register_environment('PegInsertion', PegInsertion, default_config)`. Then `registry.load('PegInsertion', config_overrides={...})` works in the trainer. Also register `PegInsertionForce` as the same class with `use_force_obs=True` default, for convenience.

### 2.3 Domain randomization (`peg_insertion/env.py::domain_randomize`) — **on by default in every run**
`register_environment(name, env_class, cfg_class)` takes no randomizer and `registry.get_domain_randomizer('PegInsertion')` would return `None` (silent no-DR). So `scripts/train.py` imports `domain_randomize` directly from `peg_insertion.env` and passes it to Brax PPO as `randomization_fn` whenever `--domain_randomization` is true, and **the flag defaults to True** in our train.py (the README describes DR as part of the method, so it must actually run). Copy the structure of a Playground `domain_randomize` (e.g. the Franka one): signature `(model: mjx.Model, rng) -> (model_v, in_axes)`, with `in_axes = jax.tree_util.tree_map(lambda x: None, model)` then the changed fields set to `0`. Randomize per env: pair friction for the 5 peg pairs ×[0.7,1.3] (`pair_friction`), peg mass ×[0.8,1.2] (`body_mass`), and the 7 arm position-actuator gains ×[0.9,1.1] — **scale `actuator_gainprm[:,0]` and `actuator_biasprm[:,1]` together** (kp and −kp), or the servo gets biased.

### 2.4 Tests (`tests/test_env.py`) — run them in the background
First-time compile of three models on sm_120 can exceed a foreground call, so always: `setsid nohup pytest tests/ -x -q > logs/pytest.log 2>&1 < /dev/null &` then poll `sleep 120 && tail -n 20 logs/pytest.log`. Reuse one compiled env across tests where possible (module-scoped fixtures).
- Env builds for clearance 2.0, 1.0, 0.5 mm; inner hole width measured from wall geom positions equals `pw + 2*clearance` to 1e-6.
- At the keyframe: world `tip_z < hand_z`; `|tip_pos − gripper_site_pos| ≈ pl − 0.01` (1e-4).
- `reset` and one `step` run under `jax.jit`; obs size is 10 without force, 16 with force; reward finite.
- 50 random steps: no NaN in `qpos`.
- Two consecutive auto-resets (through `wrap_for_brax_training(..., full_reset=True)`) give different `mocap_pos`.
- Force sensor: `|F| < 1 N` in free space (it reads the peg's own weight ≈0.5 N), `> 2 N` when the tip is pressed into the table (plain MuJoCo: IK to z below table, `mj_forward`, a few `mj_step`).

**Phase 2 done when:** `pytest tests/test_env.py` passes, `peg_insertion/assets/scene_c2.0.xml` opens and renders a PNG (save to `results/scene_preview.png`), commit "Phase 2: PegInsertion env". → Start Phase 3 immediately in this same turn.

**If it fails:** IK NaNs → check the hand orientation matrix matches Playground's `_start_tip_transform` convention; contact explosions → increase `solref` time constant on peg/walls (`solref="0.02 1"`) and lower `action_scale`; MJX compile errors about collisions → ensure every collision geom is `box` or `capsule`, none `mesh`/`cylinder`.

---

## Phase 3 — Physics sanity: scripted insertion (no learning)

**Goal:** prove the simulation can physically do the task before asking RL to learn it.

### 3.1 `scripts/scripted_insert.py`
A hand-written controller: (1) move tip to 3 cm above the *true* hole entrance, (2) descend at 1 mm/step. Run 20 episodes at each clearance (2.0, 1.0, 0.5 mm) with hole randomization on, record success rate, steps to success, peak force. Also run a version that starts 1.5 mm off-centre to see the jamming behaviour at 0.5 mm.

### 3.2 `tests/test_scripted.py`
Asserts scripted straight-down insertion succeeds ≥ 95% at 2.0 mm and ≥ 90% at 1.0 mm (centred). At 0.5 mm, just records the number (it should still be high when centred; the point of the project is that *learned* control with noisy observations is what gets hard).

### 3.3 Render one scripted episode to `results/videos/scripted_2mm.mp4` with `scripts/render.py` (uses `mujoco.Renderer` + `mediapy.write_video`, 640×480, osmesa). Record how long rendering takes.

**Done when:** tests pass, numbers logged to `results/scripted_baseline.csv`, video exists, commit "Phase 3: scripted insertion sanity". → Start Phase 4 immediately in this same turn.

**If it fails (scripted insertion fails at 2 mm):** this is a physics/geometry bug, not an RL problem. Check wall positions, peg protrusion, contact pairs, that `mocap_pos` is applied before `mj_forward`. One fix attempt, then **T5**.

---

## Phase 4 — Training pipeline and baseline at 2 mm

### 4.1 `scripts/train.py`
Copy `third_party/mujoco_playground/learning/train_jax_ppo.py` into `scripts/train.py` (keep its Apache-2.0 header; add a line noting the modifications) and edit:
- Replace the hardcoded `os.environ["MUJOCO_GL"] = "egl"` with `os.environ.setdefault("MUJOCO_GL", "osmesa")`.
- `import peg_insertion` at the top (registers the env).
- **Replace `get_rl_config(...)`** (it is a hardcoded `elif` chain in `manipulation_params.brax_ppo_config` and raises `ValueError: Unsupported env` for our name) with `from peg_insertion.configs import ppo_config`.
- **Replace the randomizer lookup** (`registry.get_domain_randomizer` returns `None` for us) with `from peg_insertion.env import domain_randomize`; `--domain_randomization` defaults to **True**.
- Pass `wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True)` so hole position and observation bias are re-sampled every episode (see 2.2).
- Add flags `--clearance_mm` and `--use_force_obs` that are folded into `config_overrides`.
- Add `--run_name`; write everything to `logs/<run_name>/` and `checkpoints/<run_name>/`.
- Keep TensorBoard logging on (`--use_tb` → `tensorboardX`); disable wandb.
- **Save params in a reload-friendly form:** after training, `brax.io.model.save_params(f"logs/{run}/params.pkl", params)` and also write `logs/<run>/network_config.json` (obs size, act size, hidden sizes) so `evaluate.py` can rebuild the policy with `ppo_networks.make_ppo_networks(..., preprocess_observations_fn=running_statistics.normalize)` → `make_inference_fn` → `load_params`.
- At the end: write `logs/<run_name>/final_metrics.json` (`success_rate` from a 256-episode rollout with the eval wrapper, `wall_time_s`, `sps`, `num_envs`, `num_timesteps`) **before** rendering any video, and wrap the video rendering in `try/except` so a renderer failure can never lose a finished run. Copy `final_metrics.json` to `results/final_metrics_<run>.json`, and `params.pkl` + `network_config.json` to `results/policies/<run>/` (each ~1 MB) so the repo ships reproducible policies (logs/ is gitignored).

### 4.2 PPO config for `PegInsertion` (in `peg_insertion/configs.py`)
`ppo_config()` must return the exact structure Brax PPO expects: the trainer does `functools.partial(network_fn, **ppo_params.network_factory)` and passes the rest to `ppo.train`. So build it from Playground's own base and override:
```python
from mujoco_playground.config import manipulation_params
def ppo_config():
  cfg = manipulation_params.brax_ppo_config('PandaPickCubeCartesian', 'jax')  # has network_factory with policy_obs_key='state', value_obs_key='state', num_eval_envs=128, num_resets_per_eval=1, max_grad_norm=1.0
  cfg.num_timesteps = 10_000_000; cfg.episode_length = 200; cfg.num_evals = 10
  cfg.network_factory.policy_hidden_layer_sizes = (256, 256)
  cfg.network_factory.value_hidden_layer_sizes = (256, 256, 256)
  # keep: num_envs=1024, batch_size=256, unroll_length=10, num_minibatches=8, num_updates_per_batch=8,
  #       discounting=0.97, learning_rate=5e-4, entropy_cost=7.5e-3, reward_scaling=0.1, normalize_observations=True, action_repeat=1
  return cfg
```
(Brax constraint: `batch_size * num_minibatches % num_envs == 0`; the halve-both OOM fallback preserves it.)

### 4.3 `scripts/train.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN="$1"; shift
mkdir -p logs/"$RUN"
setsid nohup python scripts/train.py --run_name "$RUN" "$@" > logs/"$RUN"/train.log 2>&1 < /dev/null &
echo $! > logs/"$RUN"/pid
echo "started $RUN pid $(cat logs/"$RUN"/pid)"
```
(`setsid` + `< /dev/null` detaches the job from the tool's shell so it survives the Bash call ending.) Poll with: `sleep 270 && tail -n 40 logs/<RUN>/train.log && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv`. Check a run is alive with `kill -0 $(cat logs/<RUN>/pid)`. Write `STARTED <run> <date>` to `PROGRESS.md` when launching and `FINISHED <run>` when `final_metrics.json` exists; a run that is STARTED but not FINISHED with a dead pid and no `final_metrics.json` was killed and must be relaunched with the same command.

### 4.4 Dry run (mandatory before every full run)
```bash
bash scripts/train.sh dry_2mm --clearance_mm 2.0 --num_timesteps 50000 --num_envs 256 --batch_size 64 --num_evals 1
```
The first PPO+env JIT compile on sm_120 takes **5–15 minutes**; that is normal, not a hang. Poll with `sleep 270`. **Done when** it finishes without error and writes `final_metrics.json`. Record SPS. If SPS is below ~5k with 1024 envs on the real run, consider `--impl warp` (Phase 4.6).

### 4.5 Full baseline
```bash
bash scripts/train.sh pos_2mm_s0 --clearance_mm 2.0 --seed 0
```
**Done when:** `final_metrics.json` success_rate ≥ 0.80 at 2 mm. Log wall time and SPS to `PROGRESS.md`.

**Expected ladder (judge T3 against this, not against perfection):** pos@2 mm ≈ 90–100 %, pos@1 mm ≈ 70–90 %, pos@0.5 mm ≈ 20–50 % (the 0.75 mm observation bias is designed to break position-only control at 0.5 mm), force@0.5 mm noticeably higher than pos@0.5 mm. If force@0.5 mm is *not* higher, that is still a reportable result (Phase 6 has a diagnostic run for it).

**If success < 0.80 after the run**, change **one** thing per retry, in this order, re-running each time: (a) `action_scale` 0.002→0.004 (faster approach) (b) `entropy_cost` 7.5e-3→2e-2 (c) `reward_scaling` 0.1→1.0 (d) `num_timesteps` ×2 (e) switch `reach` sigmoid to `'gaussian'` with margin 0.05. After 3 retries with no improvement → **T3**.

### 4.6 Optional throughput check with Warp
Run the dry run with `--impl warp` (needs `warp-lang`; Playground Franka configs default to `impl="warp"`, the CLI default is `jax`). If it compiles and SPS is ≥1.3× JAX, use `--impl warp` for the remaining runs and note it in `DECISIONS.md`. If it errors on sm_120, stay on JAX and log the error (do not escalate for this; it is optional).

Commit "Phase 4: training pipeline + 2 mm baseline". → Start Phase 5 immediately in this same turn.

---

## Phase 5 — Evaluation, plotting, rendering tools

### 5.1 `scripts/evaluate.py` — loads `logs/<run>/params.pkl` + `network_config.json`, rebuilds the policy (see 4.1), runs N=500 episodes (vectorised, deterministic policy) through the same `full_reset=True` wrapper with training-time randomization, and reads `succeeded`, `first_success_step`, `peak_force`, final `xy_err` from the final-step `state.info` (not from Brax's summed metrics). Outputs `results/eval_<run>.json` with success rate ± 95% CI (Wilson), mean/median steps to success (successes only), mean peak force, and a histogram of final xy error. Supports `--clearance_mm` override to test a policy trained at 2 mm on a 1 mm hole (generalisation row), and `--obs_hole_bias_std 0` for the diagnostic run.
### 5.2 `scripts/render.py` — renders K=3 episodes of a checkpoint to `results/videos/<run>.mp4` at 640×480, 25 fps. `mujoco.Renderer` has no text overlay, so draw the overlay (clearance, step, success) with `PIL.ImageDraw` on each frame before `mediapy.write_video`. Fixed camera looking at the hole plus a close-up `<camera name="closeup" .../>` added in the XML. Also produce `results/insertion.gif` from the best 0.5 mm force run: `ffmpeg -i results/videos/force_0.5mm_s0.mp4 -vf "fps=12,scale=480:-1" results/insertion.gif` (GitHub READMEs cannot inline mp4; keep the GIF under 5 MB).
### 5.3 `scripts/plot.py` — reads `results/summary.csv` and produces `results/success_vs_clearance.png` (x: clearance mm on a log or reversed axis; y: success %; two lines: position-only, position+force; error bars = CI; seeds as faint points) and `results/learning_curves.png` from TensorBoard event files (reward and success vs steps per condition).
### 5.4 Validate on the 2 mm baseline: eval, render, plot a one-point chart. Commit "Phase 5: eval/render/plot". → Start Phase 6 immediately in this same turn.

---

## Phase 6 — The experiment

Core conditions (4) × seeds (2 minimum, 3 if each run < 30 min), plus two cheap force runs so the chart has a real "position + force" line, not a single point:

| priority | run_name | clearance_mm | use_force_obs |
|---|---|---|---|
| 1 | pos_2mm_s0, pos_1mm_s0, pos_0.5mm_s0, force_0.5mm_s0 | 2.0 / 1.0 / 0.5 / 0.5 | f / f / f / t |
| 2 | pos_2mm_s1, pos_1mm_s1, pos_0.5mm_s1, force_0.5mm_s1 | same | same |
| 3 | force_2mm_s0, force_1mm_s0 | 2.0 / 1.0 | true |
| 4 (if time) | seed 2 of everything, force_2mm_s1, force_1mm_s1 | | |

`scripts/sweep.sh` runs them **sequentially** in priority order (one GPU, 8 GB) and calls `evaluate.py` + `render.py` after each, so partial results are balanced if time runs out. After each run append a row to `results/summary.csv` (`run, clearance_mm, force_obs, seed, success, ci_lo, ci_hi, steps_mean, peak_force_mean, wall_time_s, sps, num_envs, num_timesteps`) and update `PROGRESS.md`. **If behind schedule, cut in this order:** priority-4 runs → generalisation rows → learning-curve plot → seed 1 of the 1 mm condition. **Never cut the 0.5 mm pos/force pair.**

Rules:
- Reuse Phase 4's `pos_2mm_s0` only if its config equals the final Phase 4 config; if any retry in 4.5 changed a knob, rerun it.
- Same PPO config for all runs (whatever Phase 4 settled on). Changing hyperparameters per condition invalidates the comparison. If 0.5 mm needs more steps, give **all** conditions the same larger budget and note it.
- If `pos_0.5mm` is near 0% success, that is a result, not a failure. Do not tune it up individually.
- If `force_0.5mm` is not better than `pos_0.5mm`, report it honestly, and add one diagnostic run: `force_0.5mm` with `obs_hole_bias_std=0.0` to show whether the observation bias is what makes force useful. Record as an extra row.
- Extra row (cheap, informative): evaluate `pos_2mm_s0` on 1 mm and 0.5 mm holes without retraining (generalisation).
- Watch VRAM in the first minutes of each run. OOM → halve `num_envs` and `batch_size`, keep `num_timesteps`, restart, and apply the same to all remaining runs; note it.

**Done when:** `results/summary.csv` has ≥ 8 rows, `success_vs_clearance.png` and `learning_curves.png` exist, one video per condition exists. Commit "Phase 6: experiment complete". → Start Phase 7 immediately in this same turn.

---

## Phase 7 — Documentation

### 7.1 `README.md` (Griffin's portfolio face; keep it clear and honest)
Sections: title + one-sentence pitch; `results/insertion.gif` and the result chart at the top; "What this is" (3 sentences); "Why it matters" (connector insertion in electronics assembly; industry context = NVIDIA Isaac Lab's Factory peg-insert tasks and the Newton physics engine, with links to https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py and https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics — do not add company claims you have not verified yourself); "Method" (sim, control scheme, observations, reward, PPO, domain randomization, force sensing); "Results" (table from `summary.csv` + 2 sentences of interpretation + the generalisation row); "Reproduce" (Day 0 summary + `bash scripts/sweep.sh`); "Project structure"; "Hardware" (RTX 5070 Laptop, WSL2, wall-times); "Limitations and next steps" (peg rigidly attached; square peg; no sim-to-real; Isaac Lab + Newton port as next step); "Acknowledgements" (MuJoCo Playground, MuJoCo Menagerie, Brax; built with Claude Code under Griffin's direction).
### 7.2 `WRITEUP.md` — a ~1-page results write-up in Griffin's required format: a left-aligned title, then **one single paragraph**, no section headings, no em dashes, simple grammar, roughly half a page. It states the question, the setup, the numbers, and what they mean.
### 7.3 `INTERVIEW_NOTES.md` — 12–15 Q&A pairs an interviewer might ask (why PPO, why dense reward, what domain randomization does, why force helps or didn't, what you'd change, what MJX limitations you hit, how WSL2/GPU setup went) with short answers Griffin can learn.
### 7.4 `RESUME_BULLETS.md` — 3 alternative one-line resume bullets with real numbers from `summary.csv`.
### 7.5 `LICENSE` (Apache-2.0, because `scripts/train.py` derives from MuJoCo Playground; keep its header and add a `NOTICE` line), `results/environment.md` must include `uv pip freeze` output, `nvidia-smi`, versions, import time, and a per-run wall-time/SPS table. Final `PROGRESS.md` entry and `DECISIONS.md` review. Commit "Phase 7: docs". → Start Phase 8 immediately in this same turn.

---

## Phase 8 — GitHub

```bash
gh repo create peg-insertion --private --source=. --remote=origin --description "RL peg insertion vs. hole clearance with MuJoCo Playground on an RTX 5070 (WSL2)" --push
```
Verify with `gh repo view` and `git log origin/main -1` (not `--web`). Make sure `results/*.png`, `results/insertion.gif`, `results/summary.csv`, `results/final_metrics_*.json`, `results/policies/*/`, `results/videos/*.mp4` (keep total < 50 MB; downscale videos if needed) are committed, and `.venv`, `.cache`, `logs`, `checkpoints`, `third_party` are not. Final `PROGRESS.md` entry with the repo URL and a 5-line summary of results. **Then create `DONE.md`** (one paragraph: repo URL, headline numbers, date) and commit it; `DONE.md` is the signal to Griffin that the autonomous run is finished. Then end with a message to Griffin: repo link, headline numbers, what to review before making it public, and anything flagged in `DECISIONS.md` he should understand before an interview.

---

## Appendix A — Fallbacks and known pitfalls (checked 2026-10-11)

- **JAX sees no GPU on RTX 50-series:** known report (mujoco_playground issue #299, RTX 5090, "No visible GPU devices"). The user had set `LD_LIBRARY_PATH`. Keep it unset. Ensure JAX ≥ 0.6 (Blackwell builds started at 0.6.0 with CUDA 12.8). Try `jax[cuda13]` as a second option.
- **Very long first compile:** sm_120 falls back to PTX JIT; first run may take minutes. The compilation cache dir in `env.sh` makes later runs fast.
- **TF32:** `JAX_DEFAULT_MATMUL_PRECISION=highest` (Playground README "GPU precision issues").
- **OOM on 8 GB:** `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set; reduce `num_envs`, `naconmax` scales with it. Warp has its own allocator; if using `--impl warp`, be more conservative.
- **Rendering on WSL2:** EGL platform-device is not available (Mesa D3D12 path). Use `osmesa`. Expect ~0.1–0.2 s per 640×480 frame; render ≤ 3 episodes × 200 steps per video.
- **MJX collision support:** box, capsule, sphere, plane, mesh (convex) in MJX-JAX; **no** box–cylinder. Keep peg and hole as boxes.
- **Python:** JAX 0.11 dropped Python 3.11; use 3.12 (Ubuntu 24.04 default).
- **Playground `uv.lock` may be stale** relative to `pyproject.toml`; we install with `uv pip install -e`, not `uv sync`, to avoid the lock.
- **/mnt/c performance:** slow file I/O; package install is slower but training is GPU-bound. Do not move the venv out of the project folder (boundary rule).
- **Bash tool timeout:** 10 min max; always background training and poll with `sleep`.
- **Fable billing:** on some plans Fable bills to usage credits with a consent prompt in interactive mode. Escalate only on the listed triggers.

## Appendix B — Reference files worth reading before Phase 2
- `third_party/mujoco_playground/mujoco_playground/_src/mjx_env.py` (MjxEnv, State, step, get_sensor_data)
- `.../manipulation/franka_emika_panda/pick_cartesian.py` (IK delta control, workspace clipping, `_start_tip_transform`)
- `.../manipulation/franka_emika_panda/panda.py` (`get_assets`, `_post_init`), `panda_kinematics.py` (`compute_franka_ik`)
- `.../manipulation/aloha/single_peg_insertion.py` (Playground's own insertion reward using `reward_util.tolerance`)
- `.../manipulation/__init__.py` (`register_environment`), `mujoco_playground/config/manipulation_params.py` (PPO configs)
- `third_party/mujoco_playground/learning/train_jax_ppo.py` (trainer to copy)
- Isaac Lab Factory peg-insert reward (keypoint squashing, success threshold 0.04 of hole height, xy < 2.5 mm): https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py
