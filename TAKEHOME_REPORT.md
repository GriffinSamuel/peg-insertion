# Goal 5 check-in, 2026-11-15

Goals 1 through 5 are complete. This preserves the interim Goal 5 check-in, not the Goal 12 final write-up. Griffin subsequently approved continuation through Goal 13 with no cuts. A 26m51s repeatability investigation identified variability in the original GPU simulator but no validated full-run fix; see [the findings](results/takehome/repeatability_findings.md). Execution stays unchanged and the limitation is carried forward. Binary traces and checkpoints now remain on disk only; new full test traces are saved only at final 500-episode tests unless later analysis needs them.

The main learner's average success is 65.16%, compared with 61.4% for the original policy. The measured gain is 3.76 percentage points, smaller than the 7.74-point spread across practice seeds. This is a small average improvement with mixed runs, not evidence of reliable improvement. The no-camera-bias ceiling is 94.4%, so most of the available improvement remains unrealized.

## Revised instructions and Goal 4

I reread AGENTS.md, CODEX_GOALS.md and TAKEHOME_PLAN.md in full. The existing mismatch was the obsolete sampling pause and broad Goal 4 checklist; I superseded both, added local cache ignores, and committed and pushed the earlier work. No original code, baseline, test seed or success rule needed changing. Goals 1–3 were saved in commits ac61f01 and 794195c, and Goal 4 in 6bd34c6.

The learner studies a memory bank of past practice steps. Its scorekeeper is an ensemble, meaning four networks estimating the future discounted reward of a proposed correction. The corrector learns to choose corrections with high ratings. A batch contains exactly 64 fill-up and 64 learner transitions when both exist, and all 128 transitions come from the nonempty source otherwise. Sensor scaling comes from the frozen policy; correction inputs are divided by their caps. The true camera error is saved for analysis and excluded from learning inputs. The scorekeeper's raw rating is not a calibrated chance of success.

Eleven contract tests passed. The ten-episode code trial completed with finite values and an unchanged original policy; its separate test scored 0/8, retained without tuning. The counter refused episode 201. Only the main learner was built: output dimensions 2, 3 and 5 are network-ready and start at exactly zero, while movement adapters and later tricks remain deferred. Details and fixed algorithm choices are in [TAKEHOME_PROGRESS.md](TAKEHOME_PROGRESS.md), [TAKEHOME_DECISIONS.md](TAKEHOME_DECISIONS.md) and [takehome/README.md](takehome/README.md).

## Main result and mechanism checks

Each reported practice seed used exactly 40 fill-up and 160 learner episodes, collected ten at a time. Every final result used 500 test-only episodes at test seed 123. Spread means the sample standard deviation of the five final success rates, not a confidence interval. The reported point is always episode 200, never the best earlier point.

| Practice seed | Successful tests / 500 | Final success |
|---|---:|---:|
| 0 | 365 | 73.0% |
| 1 | 278 | 55.6% |
| 2 | 315 | 63.0% |
| 3 | 305 | 61.0% |
| 4 | 366 | 73.2% |
| Mean | | 65.16% |
| Spread | | 7.74 percentage points |

Source: [goal5_per_seed.csv](results/takehome/goal5_per_seed.csv), [goal5_aggregate.json](results/takehome/goal5_aggregate.json), and the five result files named there. All five runs were designated in [benchmark_manifest.json](results/takehome/benchmark_manifest.json) before observing results.

| Reference | Successful tests / 500 | Success | Check |
|---|---:|---:|---|
| Original policy, normal camera bias | 307 | 61.4% | Goal 1 passed |
| Original policy, constant camera bias removed | 472 | 94.4% | Goal 1 passed |
| Wrapper with zero correction | 302 | 60.4% | Within 3 points of floor |
| Wrapper given the true bias | 471 | 94.2% | Within 5 points of ceiling |

These use test seed 123. Removing the constant camera bias retains step-to-step noise. Supplying the true bias is a test-only reference that cheats on information; it establishes that correcting the reading can help, not that the learner can infer that correction. Sources: [baseline.json](results/takehome/baseline.json) and [mechanism_check.json](results/takehome/mechanism_check.json).

![Learning curves for all five designated practice seeds](results/takehome/goal5_learning_curves.png)

Point 0 is the original policy. Points 40, 60, 80, 100, 120, 140, 160 and 180 use 256 test episodes; point 200 uses 500, always seed 123. The curves show a sharp early deterioration in several seeds followed by recovery. No earlier peak replaces the final score. The black line is the five-seed average, not an uncertainty band.

Rebuild this figure without collecting episodes:

```bash
source scripts/env.sh && python -B takehome/plot_goal5.py
```

## Every timing repeat, including poor results

The required timing comparison needed nine independent experiments. The first designated run for each seed supplies the main result; four repeats measure timing and repeatability and never replace a main run.

| Execution group | Seed | Role | Successful tests / 500 | Success |
|---|---:|---|---:|---:|
| One process | 0 | Main | 365 | 73.0% |
| Two processes | 1 | Main | 278 | 55.6% |
| Two processes | 2 | Main | 315 | 63.0% |
| Four processes | 3 | Main | 305 | 61.0% |
| Four processes | 4 | Main | 366 | 73.2% |
| Four processes | 0 | Timing repeat | 318 | 63.6% |
| Four processes | 1 | Timing repeat | 282 | 56.4% |
| Two seeds in one process | 0 | Timing repeat | 314 | 62.8% |
| Two seeds in one process | 1 | Timing repeat | 312 | 62.4% |

Source: [goal5_all_runs.csv](results/takehome/goal5_all_runs.csv). Each row has its own 200-episode counter and saved actor, scorekeeper and step traces. Across all nine runs, the core training code hashes match. All budgets and test labels passed audit, losses and saved final arrays were finite, and final tests detected zero through-wall episodes. The frozen policy was unchanged. The five learned runs do not have identical per-episode outcomes. No tripwire fired.

Repeatability remains unresolved: repeated seeds produced different results, including seed 0 ranging from 62.8% to 73.0%. Saved first fill-up batches already differ in sensor readings and contact forces before any learning, although their sampled biases, proposed random nudges and running guesses match exactly. This narrows the discrepancy to a stage before learner updates but does not establish its cause. Do not claim deterministic replay or attribute all variability solely to practice seeds. The earlier zero-wrapper check also passed its agreed tolerance without matching every baseline success bit. Evidence: [goal5_repeatability.json](results/takehome/goal5_repeatability.json), [goal5_source_audit.json](results/takehome/goal5_source_audit.json), [goal5_run_audit.json](results/takehome/goal5_run_audit.json), and [goal5_checks.json](results/takehome/goal5_checks.json).

## Measured speed

| Setup | Completed 200-episode runs | Total wall time | Effective seconds per run |
|---|---:|---:|---:|
| One GPU process | 1 | 116.03 s | 116.03 s |
| Two GPU processes together | 2 | 132.05 s | 66.02 s |
| Four GPU processes together | 4 | 232.04 s | 58.01 s |
| Two seeds sequentially in one process | 2 | 156.03 s | 78.01 s |

A standalone run took 1.93 minutes including startup and its curve tests, or 91.12 seconds after imports/environment construction. Four processes gave the fastest observed throughput: about one completed experiment per 58 seconds, while an individual job took longer. This is a small benchmark with different seeds, not a precise controlled estimate for every future method. Future batches will provisionally use four processes; several seeds can still share a process when useful, but a combined four-process/multiple-seed schedule has not been measured. The full detached benchmark took 637.02 seconds and exited successfully. Source: [benchmark_timings.json](results/takehome/benchmark_timings.json), [goal5_benchmark.exit.json](results/takehome/goal5_benchmark.exit.json).

## Time to finish the remaining goals

These are planning estimates for laptop-awake time, not promises. They include writing, reviewing and testing code as well as simulation. Training times include the prescribed curve tests. Extra 500-episode evaluations are estimated at 0.50 minutes each. The measured main run is 1.93 minutes; complex future methods are provisionally estimated at 2.42 minutes serial. Analysis-only rows have no training runtime.

| Goal | New training runs | Extra 500-episode tests | Minutes per training run | Hours serial | Hours with four processes and helpers |
|---|---:|---:|---:|---:|---:|
| 6: budget-matched comparisons | 15 | 0 | 1.93 | 0.98 | 0.60 |
| 7: screening, combinations, confirmation | 118 likely; up to 235 | 2 | 2.42 | 6.44 | 3.60 |
| 8: scorekeeper calibration | 0; five saved-data analyses | 0 likely | n/a | 0.50 | 0.33 |
| 9: failure counts and videos | 0; about 20 videos | 1 | n/a | 1.01 | 0.55 |
| 10: three other policies | 9 | 6 | 2.42 | 0.66 | 0.40 |
| 11: eight camera conditions | 24 | 40 | 2.42 | 2.47 | 1.65 |
| 12: write-up | 0 | 0 | n/a | 0.92 | 0.65 |
| 13: final rerun and checks | 1 | 0 | 2.42 | 0.62 | 0.37 |

Likely total: **8.1 hours** with the fastest measured setup and helpers, versus **13.6 hours** serial. Conservative worst-case planning allowance: **32.5 hours**. The worst case uses the larger exploration count, twice current solo training cost, no GPU concurrency advantage, and twice estimated code/analysis time. It is not a bound on interruptions or unforeseen debugging.

The likely Goal 7 count assumes 24 new single-change settings on three seeds (72 runs), four combinations on five seeds plus two extra seeds for three promising singles (26 runs), and up to four methods on five confirmation seeds (20 runs). The conservative 235-run scenario allows up to 43 new configurations on five seeds plus 20 confirmation runs, treating two configurations as already available references. These are forecasting assumptions, not a decision to exhaust the setting cap. The written fallback can skip combinations if no change clearly helps.

Helpers can shorten routine coding, tests, tables, charts and CPU rendering while the lead designs, reviews and interprets. The estimate assumes a 25% reduction in implementation overhead, two CPU analysis workers and three rendering workers. Helpers cannot add GPU capacity; the measured GPU concurrency already accounts for sharing the one GPU. Extra test work is conservatively left serial. No confirmation seeds have been touched.

Full assumptions and reproducible calculations: [time_estimate.md](results/takehome/time_estimate.md), [time_estimate.json](results/takehome/time_estimate.json). Recompute with:

```bash
source scripts/env.sh && python -B takehome/estimate_time.py
```

## Proposed cuts, in order

None of these cuts has been applied. Savings use the faster setup and are approximate, not fully additive.

1. Skip Goal 7 combinations: about 0.90 hours saved. The existing fallback already does this if screening finds no clear winner.
2. Screen 12 new settings instead of 24: about 1.29 hours saved; Griffin would need to approve the reduced experiment list.
3. Retrain on four camera conditions instead of eight: about 0.81 hours saved, dropping the corresponding condition evaluations as well.
4. Drop the position-only and hovering-policy extensions: about 0.23 hours saved.
5. Render one example per failure type per side instead of two: about 0.15 hours saved.

## Budget, usage and resuming

One learning setting has been tried. Practice use is 1810 episodes: 1000 in the five main runs, 800 in four independent timing repeats, and 10 in the code trial. Every experiment stayed within its own budget. There have been 24,956 complete test-only episodes across preflight and Goals 1–5, plus 16 reset-only probes; none was learned from. All runs, including poor ones, are retained.

The account usage allowance is not visible: the runtime reports `rate_limits: null`, so no percentage used or remaining can be stated. Cumulative token counters include repeated/cached context and are not allowance percentages. The visible context window is 258,400 tokens; four agents can be active including the lead. No fixed session deadline is exposed. Detached processes survive the launching terminal, but not a sleeping/shut-down laptop or WSL shutdown. Progress logs, separate episode ledgers and checkpoints support resumption. Source: [goal5_usage_snapshot.json](results/takehome/goal5_usage_snapshot.json).

The lead designed and reviewed the learner, interpreted results, wrote this report and made every experimental choice. Two gpt-6-luna/low helpers handled narrow plotting and data-audit tasks in separate files; their outputs were reviewed. No fallback or success-driven tuning was used through Goal 5. PLAN GAP choices remain listed in TAKEHOME_DECISIONS.md.

**Check-in resolved:** Griffin approved continuation through Goal 13 with no cuts. Repeatability remains a documented limitation after the bounded investigation; tests, seeds and reported results are unchanged. See TAKEHOME_PROGRESS.md for the current phase.
