# Take-home implementation

Run commands from the project root inside WSL. All generated data and caches
stay under `results/takehome/`. Existing policies and environment files are read
without modification.

```bash
source scripts/env.sh
python -B takehome/baseline.py
python -B takehome/mechanism.py
python -B takehome/check_goal4.py
python -B takehome/train.py --policy force_0.5mm_s2 --seeds 0
```

The main learner is a deterministic actor-critic: the actor proposes a camera
correction, and the critic learns a discounted future reward from stored
practice transitions. It has four independently initialized critics, each with
two 64-unit layers and layer normalization. A target network is a slowly updated
copy used to provide training targets. The smallest prediction from two randomly
chosen target critics supplies the backup; the actor maximizes the mean of all
critics. Four critic updates are made per valid new learner transition, with an
actor update every fourth critic update. This adapts the ensemble and replay
ideas discussed in [REDQ](https://arxiv.org/abs/2101.05982) and
[RLPD](https://proceedings.mlr.press/v202/ball23a.html); it is not an exact
implementation of either paper.

The output layer starts at exactly zero. Exploration adds clipped Gaussian
noise only during practice. The checkpoint's sensor normalization is reused.
Physical nudges and the running estimate are divided by their caps before they
enter the critic. True bias is saved for analysis and excluded from the gradient
update inputs. The raw Q rating is not a success probability: discounting favors
earlier wins, and time limits retain the backup while true termination stops it.

Each experiment collects 10 episodes per batch, reserves them in `budget.json`
before collection, and never permits a budget above 200. A partially collected
reservation stays counted. Replay excludes transitions after learner termination
and rejects test-labelled or unreserved episodes. When both sources exist, a
128-transition training batch contains exactly 64 fill-up and 64 learner
transitions. With only one source, all 128 come from that source.

The complete per-step practice records are in `practice_*.npz`; replay contains
only valid learner transitions. Each test retains a small summary and a local
end-of-episode archive. Full step-by-step test traces are saved only for the final
500-episode test, unless a later analysis explicitly needs another trace.
All tests are labelled test-only. NPZ arrays, model checkpoints and videos stay
on disk and are ignored by git; small summaries, tables, charts, logs and code
are committed. The test runner reuses the
existing rollout's environment wrappers, seed splitting, statistics and physical
validity measurement. It accepts network parameters as dynamic arguments so
repeated curve points reuse compiled functions. Every curve checkpoint saves
the actor and critics. `checkpoint.pkl` also supports resuming after a completed
practice batch. If an interruption leaves an episode reservation without saved
data, the runner refuses to recollect it and silently exceed the budget.

The 3- and 5-output networks are shape-tested for Goal 6; movement adapters and
the later exploration tricks are deliberately deferred. The ten-episode code
trial uses seed 999 and does not count as a setting. Reported main experiments
use practice seeds 0–4 and the exact tests in `CODEX_GOALS.md`.

To start the predeclared real-run GPU benchmark detached:

```bash
source scripts/env.sh
python -B takehome/detach.py --name goal5_benchmark -- python -B takehome/benchmark.py
```

`benchmark_manifest.json` fixes which runs supply the headline results before
training. Timing repeats are retained and never replace those runs. Each run
has its own independent 200-episode ledger. Completed result files are skipped.
After the benchmark, rebuild the plot and per-seed table with:

```bash
source scripts/env.sh
python -B takehome/plot_goal5.py
```
