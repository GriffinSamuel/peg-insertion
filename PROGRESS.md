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
- STARTED pos_2mm_s0 (sweep) 2026-10-12 06:41
- FINISHED pos_2mm_s0 success=0.988 wall=5.1min sps=70866.0 2026-10-12 06:47
- STARTED pos_1mm_s0 (sweep) 2026-10-12 06:52
- FINISHED pos_1mm_s0 success=0.000 wall=5.1min sps=70694.6 2026-10-12 06:59
- STARTED pos_0.5mm_s0 (sweep) 2026-10-12 07:03
- ABORTED pos_0.5mm_s0 2026-10-12 07:05 (sweep stopped on purpose: reward fix, all runs will be redone)

## 2026-10-12 07:05 PDT — Instability diagnosis, fix attempt 2 (reward)
- lr 2e-4 sweep: pos_2mm_s0 0.988, but pos_1mm_s0 oscillated 0-0.83 during training and ended at 0.000 (deterministic eval also 0).
- Ruled out: observation-normalizer blow-up (stats sane); physics terminations/NaNs under Warp (0 of 256 episodes).
- Rollout of the failed 1 mm policy: it descends with ~2 mm lateral error, lands on the wall rim and slides outward. The reward had no mm-scale gradient (align = 20 mm-wide Gaussian, reach flat within 2 mm), so precision was only rewarded by actually falling in.
- Fix attempt 2: align term linear over 5 mm (was 20 mm Gaussian), same for all conditions. Testing seed 0 of pos 2/1/0.5 mm and force 0.5 mm in parallel (rw_* runs, 20M steps). If instability persists -> T1 escalation to fable-debugger.
- STARTED rw_pos_2mm, rw_pos_1mm, rw_pos_0.5mm, rw_force_0.5mm 2026-10-12 07:05
- FINISHED rw_* reward-fix check 2026-10-12 07:28: pos2 0.996, pos1 0.957, pos0.5 0.844, force0.5 0.980 (256-ep final eval each); stable. Final config = lr 2e-4, 20M steps, align margin 5 mm linear.
- STARTED pos_2mm_s0 (sweep) 2026-10-12 07:31
- FINISHED pos_2mm_s0 success=0.992 wall=4.8min sps=73990.7 2026-10-12 07:37
- STARTED pos_1mm_s0 (sweep) 2026-10-12 07:42
- FINISHED pos_1mm_s0 success=0.977 wall=4.9min sps=72743.0 2026-10-12 07:48
- STARTED pos_0.5mm_s0 (sweep) 2026-10-12 07:53
- FINISHED pos_0.5mm_s0 success=0.000 wall=4.8min sps=74886.0 2026-10-12 07:59
- STARTED force_0.5mm_s0 (sweep) 2026-10-12 08:03
- FINISHED force_0.5mm_s0 success=1.000 wall=4.8min sps=75349.2 2026-10-12 08:10
- STARTED pos_2mm_s1 (sweep) 2026-10-12 08:14
- FINISHED pos_2mm_s1 success=0.973 wall=5.7min sps=65399.7 2026-10-12 08:21
- STARTED pos_1mm_s1 (sweep) 2026-10-12 08:26
- FINISHED pos_1mm_s1 success=0.961 wall=5.6min sps=65495.6 2026-10-12 08:33
- STARTED pos_0.5mm_s1 (sweep) 2026-10-12 08:37
- FINISHED pos_0.5mm_s1 success=0.828 wall=5.6min sps=66178.0 2026-10-12 08:44
- STARTED force_0.5mm_s1 (sweep) 2026-10-12 08:48
- FINISHED force_0.5mm_s1 success=0.000 wall=5.3min sps=70428.0 2026-10-12 08:55

## 2026-10-12 08:56 PDT — Phase 6 sweep (final config) progress
- pos_2mm s0/s1 0.992/0.973; pos_1mm s0/s1 0.977/0.961; pos_0.5mm s0/s1 0.000/0.828; force_0.5mm s0/s1 1.000/0.000 (256-ep final_metrics; 500-ep evals in summary.csv).
- 0.5 mm is seed-dependent for both conditions; adding seeds 3-4 for the 0.5 mm pair after the sweep (DECISIONS).
- STARTED force_2mm_s0 (sweep) 2026-10-12 08:59
- FINISHED force_2mm_s0 success=0.992 wall=5.7min sps=65336.6 2026-10-12 09:07
- STARTED force_1mm_s0 (sweep) 2026-10-12 09:11
- FINISHED force_1mm_s0 success=0.996 wall=5.6min sps=65635.2 2026-10-12 09:18
- STARTED pos_2mm_s2 (sweep) 2026-10-12 09:23
- FINISHED pos_2mm_s2 success=1.000 wall=5.5min sps=67752.7 2026-10-12 09:30
- STARTED pos_1mm_s2 (sweep) 2026-10-12 09:34
- FINISHED pos_1mm_s2 success=0.000 wall=5.4min sps=69002.5 2026-10-12 09:41
- STARTED pos_0.5mm_s2 (sweep) 2026-10-12 09:45
- FINISHED pos_0.5mm_s2 success=0.000 wall=5.3min sps=70503.9 2026-10-12 09:52
- STARTED force_0.5mm_s2 (sweep) 2026-10-12 09:56
- FINISHED force_0.5mm_s2 success=0.609 wall=5.5min sps=66603.0 2026-10-12 10:03
- STARTED force_2mm_s1 (sweep) 2026-10-12 10:08
- FINISHED force_2mm_s1 success=0.957 wall=5.6min sps=65941.3 2026-10-12 10:15
- STARTED force_1mm_s1 (sweep) 2026-10-12 10:19
- FINISHED force_1mm_s1 success=0.016 wall=5.5min sps=67290.6 2026-10-12 10:26
- 2026-10-12 10:31 Main sweep (16 runs) DONE. final_metrics success: pos_2mm 0.992/0.973/1.000, pos_1mm 0.977/0.961/0.000, pos_0.5mm 0.000/0.828/0.000, force_0.5mm 1.000/0.000/0.609, force_2mm 0.992/0.957, force_1mm 0.996/0.016. Launching 10 extra replicate runs (seeds 3-4 core, s2 force 2/1 mm).
- STARTED pos_2mm_s3 (sweep) 2026-10-12 10:34
- 2026-10-12 10:35 Failure-mode rollouts (deterministic, 256 eps): pos_1mm_s2 (0%) hovers at the entrance ~4.5 mm off-centre, never pushes in; force_0.5mm_s1 (0%) hovers 8-10 mm ABOVE the hole and never touches it; pos_0.5mm_s1 (80%) descends slowly and searches, mean final depth 21 mm. No terminations/NaNs. Failed seeds are 'hover near the hole' local optima.
- FINISHED pos_2mm_s3 success=0.957 wall=5.7min sps=65131.6 2026-10-12 10:41
- STARTED pos_1mm_s3 (sweep) 2026-10-12 10:45
- FINISHED pos_1mm_s3 success=0.500 wall=5.5min sps=66810.4 2026-10-12 10:52
- STARTED pos_0.5mm_s3 (sweep) 2026-10-12 10:57
- FINISHED pos_0.5mm_s3 success=0.051 wall=5.5min sps=66895.4 2026-10-12 11:04
- STARTED force_0.5mm_s3 (sweep) 2026-10-12 11:08
- FINISHED force_0.5mm_s3 success=0.992 wall=5.6min sps=65929.3 2026-10-12 11:15
- STARTED pos_2mm_s4 (sweep) 2026-10-12 11:19
- FINISHED pos_2mm_s4 success=1.000 wall=5.5min sps=67494.9 2026-10-12 11:26
- STARTED pos_1mm_s4 (sweep) 2026-10-12 11:30
- FINISHED pos_1mm_s4 success=0.000 wall=5.5min sps=67316.5 2026-10-12 11:37
- STARTED pos_0.5mm_s4 (sweep) 2026-10-12 11:42
- FINISHED pos_0.5mm_s4 success=0.016 wall=5.6min sps=66726.1 2026-10-12 11:49
- STARTED force_0.5mm_s4 (sweep) 2026-10-12 11:53
- FINISHED force_0.5mm_s4 success=0.820 wall=5.6min sps=65825.4 2026-10-12 12:00
- STARTED force_2mm_s2 (sweep) 2026-10-12 12:04
- FINISHED force_2mm_s2 success=1.000 wall=5.8min sps=63411.9 2026-10-12 12:12
- STARTED force_1mm_s2 (sweep) 2026-10-12 12:16
- FINISHED force_1mm_s2 success=0.992 wall=5.7min sps=65417.5 2026-10-12 12:24
- 2026-10-12 12:29 Extra replicates DONE: pos_2mm s3/s4 0.957/1.000, pos_1mm s3/s4 0.500/0.000, pos_0.5mm s3/s4 0.051/0.016, force_0.5mm s3/s4 0.992/0.820, force_2mm_s2 1.000, force_1mm_s2 0.992 (final_metrics, 256 ep). Running postprocess (re-eval all with final evaluate.py, generalisation + no-bias diagnostic rows, plots).

## 2026-10-12 13:40 PDT — Phase 6 done (experiment complete)
- 26 runs, final config (warp, 1024 envs, 20M steps, lr 2e-4, align 5 mm), each evaluated on 500 fresh episodes (deterministic policy, DR on). Mean of per-seed success:
  - position only: 2 mm 98.8% (5 seeds: 100/98/100/97/100), 1 mm 48.0% (96/96/0/48/0), 0.5 mm 17.0% (0/79/0/5/2)
  - position + force: 2 mm 97.9% (3 seeds: 99/95/100), 1 mm 66.3% (99/0/99), 0.5 mm 68.8% (5 seeds: 98/0/63/98/85)
  - 0.5 mm force vs position: +51.8 points, exact one-sided permutation test over seeds p = 0.044 (5 vs 5); 4/5 force seeds vs 1/5 position seeds learned to insert (>=50%).
- No-bias diagnostic (0.5 mm policies evaluated with a perfect hole estimate): failed seeds stay at 0% (pos s0/s2/s3/s4, force s1) -> those are training failures ("hover near the hole" local optima), not bias. Successful seeds improve: pos s1 79 -> 100%, force s2 63 -> 94%, s4 85 -> 96%.
- Generalisation: pos_2mm policies (s0/s1/s2) tested without retraining: 1 mm 76.8/59.8/83.8%, 0.5 mm 31.6/25.2/47.2% -> a 2 mm-trained policy beats position-only policies trained AT 0.5 mm (17% mean); suggests a clearance curriculum.
- Validity: 0 through-wall episodes in all 26 x 500 evals; no NaN episodes. Mean steps to insert grow with tightness (pos 102 -> 134; force 95 -> 150). Mean peak contact force 41-101 N.
- Per-run wall time ~5.4 min (incl. ~20 s JIT), ~68k env steps/s. Files: results/summary.csv, summary_by_condition.csv, results_table.md, success_vs_clearance.png, learning_curves.png, videos/ (26 runs + 2 scripted), insertion.gif, policies/.
