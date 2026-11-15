# Instructions for coding agents working in this repository

You are helping Griffin with a take-home project. Read `TAKEHOME_PLAN.md` (what and why) and `CODEX_GOALS.md` (one goal per phase, with checks) before doing anything.

Rules that always apply:

1. Work only on the git branch `takehome`.
2. Never edit or delete existing code in `peg_insertion/`, `scripts/`, `tests/`, existing policies, or anything already in `results/`. New code goes in `takehome/`. New outputs go in `results/takehome/`.
3. Work through the goals in order without waiting for Griffin, except at the single check-in after Goal 5. Follow the autopilot rules at the top of `CODEX_GOALS.md`: check your own work against each goal's checks, use the written fallbacks, and halt only when a tripwire fires. Write the phase report to `TAKEHOME_PROGRESS.md` after every goal so the work can be resumed.
4. The practice budget is 200 episodes per experiment, counting every episode that is learned from. Build a counter that refuses to go past 200. Test episodes that are not learned from are separate and must be labelled as such.
5. Success is an honest measurement, not a high number. Never change the test, the test seeds, the success rule, or the budget to make a result look better. Report every run, including bad ones.
6. Log every choice not already written in the plan as one line with its reason in `TAKEHOME_DECISIONS.md`. Log what you did, with dates, in `TAKEHOME_PROGRESS.md`. Keep a running count of how many settings you have tried.
7. Run everything inside WSL. Start each shell with `source scripts/env.sh`. Reuse `peg_insertion/rollout.py` to load policies and run test episodes.
8. If something in the plan looks wrong or impossible, say so and ask. Do not quietly work around it.
9. Griffin is learning this field. Explain results in plain language and define any technical term the first time you use it.
10. Model roles. The top model is the advisor and lead: it plans each goal, designs the learner, diagnoses anything that goes wrong, reads and interprets results, reviews all code written by helpers before it is used, and writes the report. Cheaper helper agents do routine work: running experiments and sweeps, collecting results into tables, plotting, rendering videos, running tests, and simple file edits. Give each helper a narrow written task and the exact checks its output must pass. Helpers never decide what a result means, never change the test, the seeds, or the budget, and never write to `TAKEHOME_DECISIONS.md`. If a helper's output fails its checks twice, the top model does that task itself.
11. Saving time. Never sit idle while something runs: start long runs detached and use the wait to write or review the next piece. Give independent tasks to helpers in parallel. Start-up is slow here (about a minute per Python process), so run several seeds or settings inside one process where possible instead of launching one process each. Before running many GPU jobs at once, measure whether running 1, 2 or 4 together actually finishes sooner, and use the fastest. In the original project, four full training runs side by side were no faster than one after another. Parallel helpers must never write to the same file, and only the top model edits shared code.
