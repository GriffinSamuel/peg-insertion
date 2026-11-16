# Remaining-work estimate at the Goal 5 check-in

These are planning estimates based on the completed benchmark, not measured runtimes for future methods.

| Goal | New 200-episode runs | Extra 500-episode tests | Work | Minutes per training run | Hours serial | Hours with fastest setup and helpers |
|---|---:|---:|---|---:|---:|---:|
| 6 | 15 | 0 | 15 new learners; reuse main/floor/ceiling | 1.93 | 0.98 | 0.60 |
| 7 | 118 | 2 | 72 screening + 26 follow-up + 20 confirmation; upper bound 235 | 2.42 | 6.44 | 3.60 |
| 8 | 0 | 0 | 5 saved-data analyses; no fresh practice | n/a | 0.50 | 0.33 |
| 9 | 0 | 1 | 20 videos at 1.5 min each; one before-trace test | n/a | 1.01 | 0.55 |
| 10 | 9 | 6 | 3 policies x 3 seeds; six floor/ceiling tests | 2.42 | 0.66 | 0.40 |
| 11 | 24 | 40 | 8 problems x 3 retrains; 24 transfer + 8 floor + 8 oracle tests | 2.42 | 2.47 | 1.65 |
| 12 | 0 | 0 | Report and figure reproduction; no new experiment | n/a | 0.92 | 0.65 |
| 13 | 1 | 0 | 1 confirmation-seed rerun plus old/new tests and audit | 2.42 | 0.62 | 0.37 |

Extra 500-episode tests are estimated at 0.50 minutes each, including a startup allowance. Training-run times include their prescribed learning-curve tests. Analysis-only rows include code, review and CPU work.

Likely total: 13.6 hours serial, or 8.1 hours using the fastest measured setup and helpers. Conservative worst-case planning allowance: 32.5 hours.

## Cut list (requires a plan change at the check-in)

1. Skip Goal 7 combination round: about 0.90 hours saved.
2. Screen 12 new settings instead of 24: about 1.29 hours saved.
3. Retrain on four camera problems instead of eight: about 0.81 hours saved.
4. Drop the position-only and hovering-policy extensions: about 0.23 hours saved.
5. Render one example per failure type per side instead of two: about 0.15 hours saved.

## Assumptions

- Estimates, not runtime guarantees; laptop must remain awake.
- Likely Goal 7: 24 new single-change settings x 3 seeds, four combinations x 5 plus two extra seeds for three promising singles, then up to four methods x 5 confirmation seeds.
- Goal 7 upper bound: up to 43 new configurations x 5 seeds plus 20 confirmation runs, treating two of the 45 configurations as already available references.
- Future complex learners estimated at 1.25 times main-run cost; worst case allows twice solo cost and no GPU concurrency advantage.
- Writing/review/plot overhead is a planning estimate; helpers assumed to shorten that portion by 25%. They do not multiply GPU capacity.
- Saved-data analyses and rendering estimates are not benchmarked here; videos assume 20 x 1.5 min and three CPU workers.
- GPU throughput comparisons use small groups and different seeds, so the fastest observed setup is provisional. All forecasts include implementation/review time and leave evaluation work serial.
- Worst case is a planning allowance, not an upper bound on interruptions or debugging. No confirmation seeds have been used for this estimate.
- Cuts are proposals for Griffin, not changes already made. Savings are approximate and not all independent.
