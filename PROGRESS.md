# PROGRESS

Deadline: 2026-10-19 (start 2026-10-12 + 7 days; checked for trigger T7)

## 2026-10-12 03:42 PDT — Phase 1.1 start
- Read CLAUDE.md and PLAN.md. Day 0 verified: driver 596.58 (CUDA 13.2), RTX 5070 Laptop 8151 MiB visible via nvidia-smi, 19 GiB RAM + 8 GiB swap in WSL, 397 GB free on C:.
- Fable availability check: fable-debugger subagent responded "OK" (Fable available).
- Created folder layout, PROGRESS.md, DECISIONS.md; initialized git.

## 2026-10-12 04:15 PDT — Phase 1 done
- Venv created in-project; one-resolve install (Playground editable + jax[cuda12] + extras): jax 0.11.2, mujoco/mjx 3.15.0, brax 0.14.2, warp-lang 1.18.0.
- `jax.default_backend()` = gpu, `[CudaDevice(id=0)]` on first try (no troubleshooting needed).
- Menagerie cloned (pinned commit 1b86ece) -> MENAGERIE OK. Playground smoke test -> SMOKE OK (first step incl. compile 20.9 s).
- osmesa rendering works. Import time 36-60 s (acceptable; batching probes into scripts).
- Details in results/environment.md. Next: Phase 2 (env). Draft xml_builder.py / env.py / tests already written while installs ran.

## 2026-10-12 04:40 PDT — Phase 2 done (PegInsertion env)
- `peg_insertion/xml_builder.py` (scene: Panda + rigid peg 20x20x60 mm, table, 4-wall hole, 5 explicit contact pairs, wrist F/T sensor), `peg_insertion/env.py` (MjxEnv, IK delta-position control, biased hole obs, reward per PLAN, domain_randomize), registered as `PegInsertion` / `PegInsertionForce`.
- Two physics fixes found by probing (see DECISIONS): gravity compensation (arm sagged 23 mm without it) and an exactly-vertical, axis-aligned gripper orientation.
- `pytest tests/` -> 12 passed in 7.3 min (hole width = pw + 2c to 1e-6 for 2/1/0.5 mm; peg below hand, |tip-TCP| = 50 mm; obs 10/16; 50 random steps no NaN; full_reset re-samples the hole on each auto-reset; |F| = 0.49 N free, > 2 N pressed).
- Scene preview: results/scene_preview.png. Generated XMLs: peg_insertion/assets/scene_c{2.0,1.0,0.5}.xml.
- Next: Phase 3 scripted baseline CSV + video (scripted tests already pass: 100% centred at 2/1/0.5 mm).
