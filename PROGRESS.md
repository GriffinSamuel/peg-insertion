# PROGRESS

Deadline: 2026-10-19 (start 2026-10-12 + 7 days; checked for trigger T7)

## 2026-10-12 03:42 PDT — Phase 1.1 start
- Read CLAUDE.md and PLAN.md. Day 0 verified: driver 596.58 (CUDA 13.2), RTX 5070 Laptop 8151 MiB visible via nvidia-smi, 19 GiB RAM + 8 GiB swap in WSL, 397 GB free on C:.
- Fable availability check: fable-debugger subagent responded "OK" (Fable available).
- Created folder layout, PROGRESS.md, DECISIONS.md; initialized git.

## 2026-10-12 04:01 PDT — Phase 1 done
- Venv created in-project; one-resolve install (Playground editable + jax[cuda12] + extras): jax 0.11.2, mujoco/mjx 3.15.0, brax 0.14.2, warp-lang 1.18.0.
- `jax.default_backend()` = gpu, `[CudaDevice(id=0)]` on first try (no troubleshooting needed).
- Menagerie cloned (pinned commit 1b86ece) -> MENAGERIE OK. Playground smoke test -> SMOKE OK (first step incl. compile 20.9 s).
- osmesa rendering works. Import time 36-60 s (acceptable; batching probes into scripts).
- Details in results/environment.md. Next: Phase 2 (env). Draft xml_builder.py / env.py / tests already written while installs ran.

## 2026-10-12 04:37 PDT — Phase 2 done (PegInsertion env)
- `peg_insertion/xml_builder.py` (scene: Panda + rigid peg 20x20x60 mm, table, 4-wall hole, 5 explicit contact pairs, wrist F/T sensor), `peg_insertion/env.py` (MjxEnv, IK delta-position control, biased hole obs, reward per PLAN, domain_randomize), registered as `PegInsertion` / `PegInsertionForce`.
- Two physics fixes found by probing (see DECISIONS): gravity compensation (arm sagged 23 mm without it) and an exactly-vertical, axis-aligned gripper orientation.
- `pytest tests/` -> 12 passed in 7.3 min (hole width = pw + 2c to 1e-6 for 2/1/0.5 mm; peg below hand, |tip-TCP| = 50 mm; obs 10/16; 50 random steps no NaN; full_reset re-samples the hole on each auto-reset; |F| = 0.49 N free, > 2 N pressed).
- Scene preview: results/scene_preview.png. Generated XMLs: peg_insertion/assets/scene_c{2.0,1.0,0.5}.xml.
- Next: Phase 3 scripted baseline CSV + video (scripted tests already pass: 100% centred at 2/1/0.5 mm).

## 2026-10-12 04:52 PDT — Phase 3 done (scripted physics sanity)
- `scripts/scripted_insert.py` (true hole position; approach 3 cm above, descend 1 mm/step), 20 episodes per row, hole randomized:
  - centred: 100% success at 2.0 / 1.0 / 0.5 mm, success at step ~100, peak |F| ~44 N.
  - 1.5 mm off-centre: 100% at 2.0 mm (offset < clearance), 0% at 1.0 and 0.5 mm (peg lands on the wall rim and jams, peak ~65 N), as expected.
- `tests/test_scripted.py` passes (>=95% at 2 mm, >=90% at 1 mm). CSV: results/scripted_baseline.csv.
- Videos: results/videos/scripted_2mm.mp4, scripted_0.5mm.mp4 (osmesa, ~0.3 s/frame incl. overview inset, 101 frames in ~30 s).
- Next: Phase 4 training pipeline (train.py/evaluate.py/render.py/plot.py drafted), dry run.
- STARTED dry_2mm 2026-10-12 04:52 (dry run)
- STARTED pos_2mm_s0 2026-10-12 05:36 (Phase 4.5 baseline, warp, 10M steps)

## 2026-10-12 05:37 PDT — Phase 4 in progress
- train.py (derived from Playground's train_jax_ppo.py), train.sh, eval.sh, evaluate.py, render.py, plot.py, sweep.sh written.
- Fixed: JAX 0.11 removed `jax.device_put_replicated` (Brax still uses it) -> compat shim; env step now merges wrapper metrics.
- Dry run (`dry_2mm`, 50k steps, 256 envs) passed end to end: compile+train 2.1 min, final_metrics.json + video written.
- Throughput benchmark (0.5M steps, 1024 envs): JAX ~2.5k steps/s, Warp ~11k steps/s -> using Warp (tests pass under both). VRAM ~0.7 GB.
- FINISHED pos_2mm_s0 2026-10-12 05:42 success=0.992 (254/256) wall=2.9min sps=64.7k

## 2026-10-12 05:48 PDT — Phase 4 + 5 done
- Baseline `pos_2mm_s0` (warp, 1024 envs, 10M steps, PLAN 4.2 PPO config, no retries needed): final_metrics success 0.992 (254/256); 500-episode eval 0.994 (CI 0.983-0.998), mean 116 steps to success, mean peak |F| 63 N. Wall time 2.9 min (JIT 20 s with warm cache), ~65k steps/s.
- Learning curve is non-monotonic (eval success 0.52 at 1.1M, dips to 0.02-0.15, then 0.97-0.99 from 7.9M).
- Phase 5 tools validated on it: results/eval_pos_2mm_s0.json, results/videos/pos_2mm_s0.mp4 (3 episodes, 70 s render), results/success_vs_clearance.png, results/learning_curves.png, results/summary.csv.
- Runs take ~3 min, so Phase 6 will use 3 seeds for every condition (PLAN: "3 if each run < 30 min").
- STARTED pos_1mm_s0 (sweep) 2026-10-12 05:48
- FINISHED pos_1mm_s0 success=0.984 wall=3.5min sps=56200.0 2026-10-12 05:53
- STARTED pos_0.5mm_s0 (sweep) 2026-10-12 05:58
- FINISHED pos_0.5mm_s0 success=0.000 wall=3.5min sps=57343.5 2026-10-12 06:03
- STARTED force_0.5mm_s0 (sweep) 2026-10-12 06:08
- FINISHED force_0.5mm_s0 success=0.000 wall=3.6min sps=55793.4 2026-10-12 06:13

## 2026-10-12 06:19 PDT — Phase 6 priority-1 results at 10M steps + instability diagnosis
- pos_1mm_s0 0.972 (500-ep eval), pos_0.5mm_s0 0.000, force_0.5mm_s0 0.000 (final checkpoint).
- BUT force_0.5mm_s0 reached 0.82-0.98 eval success at 2.3-4.5M steps and then collapsed to 0 (eval reward 32 -> 20): PPO instability, not a property of the task. pos_2mm_s0 showed a similar transient dip. Reporting the final checkpoint now would be misleading.
- Diagnosing with single-knob runs on force@0.5mm, 20M steps, run in parallel: A = same config seed 1; B = learning_rate 2e-4; C = ADAPTIVE_KL lr schedule (desired KL 0.01). Any fix will be applied to ALL conditions and all runs redone.
- STARTED diagA_f05_seed1, diagB_f05_lr2e-4, diagC_f05_akl 2026-10-12 06:19
- Diagnostics finished 2026-10-12 06:40: A seed1@5e-4 0.992 (stable), B seed0@2e-4 0.941 final (0.94-1.0 from 2M on), C ADAPTIVE_KL 0.000. Adopted lr 2e-4 + 20M steps for all conditions; pytest (warp) 12 passed; dry run OK. Launching full sweep.
