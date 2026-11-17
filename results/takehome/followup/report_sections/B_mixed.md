## Mixed camera error: calibration plus per-attempt variation

These conditions model a calibration component that stays fixed for one run plus an independent component drawn for each attempt. The stated standard deviations are used exactly: fixed/per-attempt values of 0.65/0.375 mm, 0.53/0.53 mm, and 0.375/0.65 mm. The fixed-only condition isolates calibration error; none of these is designated the realistic condition. Each calibration seed has its own 500-test floor and zero-error reference. The unchanged within-episode camera jitter remains present.

| Model | Run-specific calibration spread, per axis | Per-attempt spread, per axis |
| --- | --- | --- |
| Normal | 0 mm | 0.75 mm |
| Fixed for the whole run | 0.75 mm | 0 mm |
| Mostly fixed | 0.65 mm | 0.375 mm |
| Even | 0.53 mm | 0.53 mm |
| Mostly per-attempt | 0.375 mm | 0.65 mm |

The run-specific draw is shared by that experiment's practice and test episodes. The per-attempt draw changes between episodes. All five models retain the original 0.2 mm noise on individual readings. The mixed values use the plan's rounded numbers exactly, without rescaling.

All 12 requested settings finished. Five three-seed gains qualified for extension, and all five retained HELPED: plain aim in the mostly-fixed and even splits, and plain movement in all three splits. No heat-map or joint addition qualified for extension. The mostly-fixed aim result clears its spread threshold by only 0.02 percentage points, so its HELPED label is especially sensitive to small result variation. The heat map collapsed on one seed in both the even and mostly-per-attempt splits; even-split joint learning also collapsed below half its matched movement base. These are UNSTABLE results, not missing runs.

| Split | Method | Seeds | Final mean ± spread | Verdict | Aim error at start / success (mm) | Paired change toward truth (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| Mostly fixed | Plain aim | 5 | 87.08% ± 4.73 points | HELPED | 0.880 / 1.263 | -0.416 |
| Mostly fixed | Heat map | 3 | 84.07% ± 3.64 points | NO REAL DIFFERENCE | 1.553 / 0.745 | +0.787 |
| Mostly fixed | Plain movement | 5 | 90.32% ± 5.04 points | HELPED | n/a | n/a |
| Mostly fixed | Aim + movement | 3 | 83.53% ± 22.30 points | NO REAL DIFFERENCE | 0.897 / 1.189 | -0.334 |
| Even | Plain aim | 5 | 79.88% ± 6.78 points | HELPED | 0.895 / 1.160 | -0.346 |
| Even | Heat map | 3 | 43.40% ± 38.49 points | UNSTABLE | 1.388 / 1.797 | -0.179 |
| Even | Plain movement | 5 | 87.64% ± 7.46 points | HELPED | n/a | n/a |
| Even | Aim + movement | 3 | 73.00% ± 25.42 points | UNSTABLE | 0.904 / 1.177 | -0.374 |
| Mostly per-attempt | Plain aim | 3 | 69.93% ± 6.52 points | NO REAL DIFFERENCE | 0.922 / 1.291 | -0.516 |
| Mostly per-attempt | Heat map | 3 | 37.67% ± 31.56 points | UNSTABLE | 1.277 / 1.982 | -0.797 |
| Mostly per-attempt | Plain movement | 5 | 88.16% ± 6.66 points | HELPED | n/a | n/a |
| Mostly per-attempt | Aim + movement | 3 | 81.27% ± 7.91 points | NO REAL DIFFERENCE | 0.922 / 1.276 | -0.440 |

Plain aim's better insertion results do not establish that it found the camera error. Its paired success-time guess error became worse in all three mixed splits. The mostly-fixed heat map did bring the guess closer to the truth, but did not improve success over matched plain aim. Success-time errors exclude unsuccessful episodes; in even-split heat-map seed 1 there was no success, so that seed has no defined success-time error or insertion time. The paired column compares each successful episode with its own initial guess, and a positive value means closer.

Recovered lost success means (method success minus floor)/(zero-error reference minus floor). It can be negative or exceed 100%; it is not clipped. The following comparison uses the same initial three seeds for every method and split, so extending only promising methods does not change the group being compared. The accompanying CSV also gives every seed and the full three- or five-seed aggregates.

| Calibration share | Plain aim | Heat map | Plain movement | Aim + movement |
| --- | --- | --- | --- | --- |
| Mostly per-attempt | 24.1% | -78.1% | 85.7% | 59.9% |
| Even | 54.5% | -57.3% | 88.6% | 34.6% |
| Mostly fixed | 80.1% | 68.8% | 97.5% | 67.1% |

On these shared three seeds, aim learning recovers more of the lost success as the calibration component grows. Movement stays strong across all three splits. This is a small calibration sample: the full five-seed movement averages are 90.32%, 87.64% and 88.16% from mostly fixed to mostly per-attempt, so the full averages are not a strictly ordered trend. Heat-map and joint additions did not beat their matched plain bases.

![Mixed-condition comparison on shared screening seeds](results/takehome/followup/B_mixed/recovery.png)

[Recovery data, including every seed](results/takehome/followup/B_mixed/recovery.csv). Thirty-four additional 500-test checks on diagnostic seed 789 found no wall-passing or invalid-number flags; the 170 inspected examples had bounded controls. A saved five-episode trajectory plot shows descent into the opening without a flagged sideways passage. These checks do not prove physical realism and were never used to choose a method. The mixed comparison consumed 9,200 practice episodes across 46 independent experiments, with exactly 200 each.
