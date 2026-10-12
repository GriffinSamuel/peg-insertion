# DECISIONS

One line per decision not settled by PLAN.md, with the reason.

- 2026-10-12: Escalations use `fable-debugger` (availability check returned OK), `opus-debugger` only as fallback.
- 2026-10-12: Cloned menagerie with plain `git clone` + checkout of Playground's pinned SHA (identical to `ensure_menagerie_exists()`), in the background in parallel with the pip install, to save time.
- 2026-10-12: Probe scripts that are not part of the pipeline live in `scripts/dev/` (kept for transparency).
- 2026-10-12: Gravity compensation (`gravcomp="1"`) on all Panda arm/hand bodies. Menagerie's soft servos (kp 300-1000) sagged ~23 mm under gravity, which made mm-level insertion impossible; real Franka controllers compensate gravity. Peg itself is not compensated (wrist sensor still feels its 0.49 N weight). Sag now ~0.2 mm.
- 2026-10-12: Fixed gripper orientation = straight down with hand x snapped to a world axis (Playground's `home` keyframe is tilted ~7 deg, which would make a square peg unable to enter a square hole). Peg faces verified parallel to the hole walls.
- 2026-10-12: Anti-windup limit `max_servo_error` = 3 mm on all three axes (PLAN said 5 mm on z only). Measured contact stiffness is ~12 N/mm, so 5 mm let the scripted controller hit 129 N peaks; 2 mm cut peaks to 35 N but slowed free motion (servo lag ~3 mm at 1 mm/step) so success came at step ~141/200; 3 mm gives ~42 N peaks and success at step ~100.
- 2026-10-12: Hole-position observation bias applied to x,y only (z bias does not change the lateral alignment problem the experiment is about).
- 2026-10-12: Scripted controller drives the commanded setpoint (plus a slow 0.2-gain xy integral on measured error while descending), not the measured tip: integrating measured error with the servo's 3-step lag caused a limit cycle.
- 2026-10-12: Torque observation normalised by 2 N*m (force by 20 N per PLAN).
- 2026-10-12: Added `peg_insertion/jax_compat.py`: Brax 0.14.2 (latest) calls `jax.device_put_replicated`, removed in JAX 0.11 (which Blackwell needs). Shim uses JAX's documented drop-in (stack + shard over devices). Single GPU, so behaviour is identical.
- 2026-10-12: Training uses MJX `impl=warp` (installed `mujoco-warp==3.15.0`, matching mujoco 3.15). Benchmark at 1024 envs: JAX ~2.5k steps/s steady, Warp ~11k steps/s (4.4x, > the 1.3x bar in PLAN 4.6). Full test suite passes under both (`PEG_IMPL=warp pytest tests/` 12 passed, scripted 0.5 mm 100%). Eval/render use the same impl as training.
- 2026-10-12: PPO learning rate 5e-4 -> 2e-4 and budget 10M -> 20M steps (21 evals) for ALL conditions; all runs redone with this one config (old 10M results archived in results/archive/lr5e-4_10M/, not deleted). Evidence (single-knob, force@0.5mm): seed 0 at 5e-4 hit 98% at 3.4M then collapsed to 0% (eval reward 32 -> 20); seed 0 at 2e-4 held 94-100% from 2M to 20M; seed 1 at 5e-4 happened to stay stable (99%); ADAPTIVE_KL schedule never learned (0%). 20M because a lower LR learns slower and runs only take ~6 min.
- 2026-10-12: 3 seeds per condition (runs are ~6 min, PLAN allows 3 when < 30 min), plus force@2mm and force@1mm (2 seeds each) for a full force line.
