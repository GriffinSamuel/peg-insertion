# peg-insertion — project brief for Claude Code

You are building a reinforcement-learning research project autonomously for Griffin, a first/second-year engineering student applying to NVIDIA Ignite and Microsoft Explore. Read this file fully. Then read `PLAN.md` and follow it phase by phase. `PLAN.md` is the source of truth for *what* to do; this file is the source of truth for *how* to behave.

## The project in one paragraph

Train a simulated Franka Panda arm to insert a square peg (already held in its gripper) into a square hole, using MuJoCo Playground (MJX, Brax PPO) on Griffin's RTX 5070 Laptop GPU inside WSL2. Measure how insertion success rate falls as the hole clearance tightens (2 mm, 1 mm, 0.5 mm), and whether adding a wrist force sensor to the observations recovers success at 0.5 mm. Deliver code, a results chart, rollout videos, a README, a short write-up, and a GitHub repo.

## Hard boundaries (never break these)

1. **Stay inside the project folder.** The project root is `/mnt/c/Users/Griff/Projects/peg-insertion` (Windows: `C:\Users\Griff\Projects\peg-insertion`). Never create, edit, move or delete any file outside it. Caches, virtualenv, logs, downloads and checkpoints all live inside it (the env vars in `scripts/env.sh` enforce this). A PreToolUse hook blocks file edits outside the root; do not try to work around it.
2. **No `sudo`, no `apt`, no system changes.** Everything you need system-wide was installed by Griffin in Day 0 of `PLAN.md`. If something system-level is genuinely missing, write it to `BLOCKED.md` and stop (see "Stopping").
3. **No destructive git.** Never `git push --force`, never rewrite history, never delete branches. Commit often on `main`.
4. **GitHub:** create the repo as **private** at the end (Phase 8). Griffin will flip it to public after he reviews it. Never post anywhere else.
5. **Never fabricate results.** If a configuration fails to train, report the failure with the actual numbers. A negative result honestly reported is a valid outcome of this project.
6. **No pausing for approval** except for the "big issue" conditions under "Stopping." Griffin asked for a hands-off run. Make reasonable decisions yourself, record them in `DECISIONS.md`, and keep going.
7. **Never end your turn while work remains.** The only two legitimate ends of a turn are: `DONE.md` exists, or `BLOCKED.md` exists. Ending a turn for any other reason is a project failure, because nobody is watching and the run idles until Griffin's next check (up to 12 hours lost). Specifically, do not: end a turn after finishing a phase (begin the next phase in the same turn); end a turn while a background run is alive (poll it with `sleep 270 && tail -n 40 logs/<run>/train.log` until `final_metrics.json` exists, then continue); write a status summary to chat and wait (status goes in `PROGRESS.md`, not chat); ask "shall I continue?", "let me know if…" or any question (decide, log in `DECISIONS.md`, continue). If you catch yourself about to write a final message, check: does `DONE.md` or `BLOCKED.md` exist? If not, you are not finished; run the next step instead.

## Working style

- **Log as you go.** After every phase (and after every training run), append a dated entry to `PROGRESS.md`: what you did, what worked, numbers, what's next. Griffin reads this file to see where things stand. Keep entries short and factual.
- **Record decisions.** Any time you choose between options not settled by `PLAN.md` (a hyperparameter, a reward weight, a workaround), add one line to `DECISIONS.md` with the reason.
- **Commit after each phase** with a clear message. Also commit before and after any risky change.
- **Run training in the background.** Bash calls time out at 10 minutes. Launch long jobs with `setsid nohup ... > logs/<run>/train.log 2>&1 < /dev/null &` (use `scripts/train.sh`), then poll with `sleep 270 && tail -n 40 logs/<run>/train.log`. Never sit in a foreground command longer than a few minutes. First-time JIT compiles on this GPU take 5–15 minutes; that is not a hang.
- **Never end your turn while work remains** (Hard boundary 7). If you notice the context was compacted or the session was resumed, first re-read `PLAN.md`, the last 60 lines of `PROGRESS.md`, `DECISIONS.md`, and `ls logs/*/pid` before acting. Any run marked `STARTED` but not `FINISHED` in `PROGRESS.md` whose pid is dead and whose `logs/<run>/final_metrics.json` is missing was killed: relaunch it with the same command.
- **Every Bash call that runs `python`, `uv` or `pytest` starts with** `cd /mnt/c/Users/Griff/Projects/peg-insertion && source scripts/env.sh && ` (env vars and the venv do not persist between calls).
- **Smoke test before every long run.** Always run the scripted-insertion test (`pytest tests/`) and a short training dry run (`--num_timesteps 50000`, see PLAN 4.4; the first one compiles for 5–15 min) before launching a full run. Never launch a multi-hour run on untested code.
- **One change at a time** when debugging training. Change one knob, rerun, compare.
- **Prefer the project's own scripts** (`scripts/*.sh`) over ad-hoc commands so runs are reproducible.
- **Keep secrets out of the repo.** Never commit `.env` files, tokens, or `~/.config` contents. `.gitignore` is set up; respect it.
- **Write for Griffin.** README and write-up must be understandable by a first-year engineering student and must honestly describe that the code was built with AI assistance under his direction. Griffin will present this in interviews, so every design choice needs a one-line "why" he can repeat.

## Fable escalation (when and how)

You run on Opus by default. A stronger model, Fable, is available as the subagent `fable-debugger` (defined in `.claude/agents/fable-debugger.md`). Fable is expensive; use it exactly when one of these triggers fires, not before and not for routine work:

**Triggers**
- T1. The same error (or same failure mode) survives **2** of your own fix attempts.
- T2. JAX does not see the GPU (`jax.default_backend()` != `"gpu"`) after you have worked through the "If backend is cpu" list in Phase 1 step 1.4 once.
- T3. Baseline training at 2 mm clearance shows **no improvement in success rate after 3 full runs** with different single-knob changes.
- T4. NaNs or exploding losses that survive one fix attempt.
- T5. Any physics result that contradicts the sanity checks in Phase 3 (e.g. scripted straight-down insertion fails at 2 mm) after one fix attempt.
- T6. You are about to change the experiment design (e.g. drop the 0.5 mm condition, change the control scheme) — get Fable's opinion first.
- T7. You estimate remaining work no longer fits the 7-day budget.

**How to escalate**
1. Write `ESCALATION.md` with: the goal of the current phase, the exact error/observation (paste logs, max 200 lines), everything you have tried with results, the relevant file paths, your current hypothesis, and the specific question you want answered.
2. Invoke the `fable-debugger` subagent and tell it to read `ESCALATION.md`, `CLAUDE.md`, `PLAN.md`, `PROGRESS.md`, and the files you list. Ask for: a diagnosis, a concrete fix (code or commands), and a way to verify the fix.
3. Apply the fix yourself, verify, and record the outcome in `PROGRESS.md` and `DECISIONS.md`. Append the resolved escalation to `ESCALATIONS_LOG.md`.
4. If Fable's fix also fails, escalate **once more** with the new information. After two Fable rounds on the same problem with no progress, treat it as a big issue (below).

If the `fable-debugger` agent fails to start because Fable is not available on this plan (or Griffin's start prompt says so), use the `opus-debugger` agent instead with the same `ESCALATION.md`, and note that in `PROGRESS.md`.

## Stopping ("big issue" conditions)

Stop work, write `BLOCKED.md` describing the problem, what you tried, and what Griffin needs to do, commit, and end your turn with a clear message to Griffin, **only** when:
- The GPU cannot be used after Phase 1 troubleshooting plus one Fable escalation (likely needs a Windows-side driver or WSL fix only Griffin can do).
- Something requires `sudo`/`apt`/Windows-side action.
- Two Fable rounds on the same **infrastructure** blocker produced no progress.
- Disk space, memory, or WSL itself is failing.
- You would need to leave the project folder to continue.

**Exception:** if the unresolved problem is *training performance* (a condition will not learn, T3/T5 after two Fable rounds), that is a scientific result, not a blocker. Record it as a negative result with the real numbers, run the remaining conditions anyway, and finish Phases 7–8. Never stop the project because one condition did not learn.

Everything else: decide, log, continue.

## Environment facts (so you don't rediscover them)

- Windows laptop: Intel Core Ultra 9 285H, 32 GB RAM, **NVIDIA RTX 5070 Laptop (8 GB VRAM, Blackwell, sm_120)**.
- WSL2 Ubuntu 24.04, Python 3.12. The project folder is on the Windows drive (`/mnt/c/...`), so file I/O is slower than native Linux. Expect `uv sync` to take longer; this is fine.
- Do **not** install any Linux NVIDIA driver inside WSL. The Windows driver provides `libcuda.so` at `/usr/lib/wsl/lib/`. `nvidia-smi` works in WSL with a limited feature set.
- Do **not** set `LD_LIBRARY_PATH`. JAX uses the pip-installed `nvidia-*` CUDA wheels. Setting it causes "No visible GPU devices."
- Blackwell needs JAX ≥ 0.6 (first JAX built with CUDA 12.8 / `compute_120`). First GPU runs JIT-compile PTX for sm_120 and are slow; the JAX compilation cache (set in `scripts/env.sh`) fixes this after the first run.
- Set `JAX_DEFAULT_MATMUL_PRECISION=highest` (TF32 matmul hurts RL stability on RTX cards; this is in `scripts/env.sh`).
- 8 GB VRAM: keep `XLA_PYTHON_CLIENT_PREALLOCATE=false`, start with `num_envs=1024`, scale `naconmax` with `num_envs`. If OOM, halve `num_envs` and `batch_size` together.
- **Rendering:** EGL does not work on WSL2. Use `MUJOCO_GL=osmesa` (software rendering; slow but fine for a few 640x480 videos). Playground's `learning/train_jax_ppo.py` hardcodes `MUJOCO_GL=egl` before imports, which is why we use our own `scripts/train.py` instead of calling it directly.
- Playground's Franka Cartesian env (`pick_cartesian.py`) uses analytic IK with delta-position actions and a fixed gripper orientation. Our env copies that scheme. There is no operational-space controller in Playground. The IK works in the TCP frame (`gripper` site, 0.100 m along hand +z); the peg tip is `pl − 0.01` further along +z.
- Playground's `wrap_for_brax_training` defaults to `full_reset=False`, which never re-samples `info` on auto-reset. Always pass `full_reset=True` or per-episode randomization silently does not happen.
- Brax's EvalWrapper **sums** `state.metrics` over an episode. Put only summable per-step quantities in `metrics`; read episode-level stats from `state.info` in your own eval rollout.
- `register_environment(name, env_class, cfg_class)` takes no randomizer, and `manipulation_params.brax_ppo_config` is a hardcoded `elif` chain: our `scripts/train.py` must import `ppo_config` and `domain_randomize` from `peg_insertion` directly.
- MJX-JAX supports box-box collisions (boxes become convex meshes). Peg = one box; hole = four box walls on a table. No mesh collision geoms.
- Menagerie assets are cloned by `git` on first env load (network needed once).
- Brax PPO ballparks on an A100: PandaPickCubeCartesian 5M steps ≈ 2 min. Expect the 5070 Laptop to be several times slower and memory-limited; plan 10–40 min per run, and measure on the first run.

## Definitions of done (summary; details in PLAN.md)

- `pytest tests/` passes (env builds, shapes correct, scripted insertion succeeds at 2 mm and 1 mm, clearance parameter actually changes geometry, force sensor reads nonzero on contact).
- At least 2 seeds trained for each of 4 conditions: `pos@2mm`, `pos@1mm`, `pos@0.5mm`, `force@0.5mm`.
- `results/summary.csv` and `results/success_vs_clearance.png` exist and match the logs.
- At least one rollout video per condition in `results/videos/`.
- `README.md` (with `results/insertion.gif`), `WRITEUP.md`, `INTERVIEW_NOTES.md`, `RESUME_BULLETS.md`, `LICENSE`, `PROGRESS.md`, `DECISIONS.md` complete; repo pushed to GitHub (private); `DONE.md` created last.
