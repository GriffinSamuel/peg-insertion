## Phase A: why the aim guess failed

These are saved final tests from practice seeds 100 to 104, with 500 test-only episodes per seed at test seed 456. They are diagnostic evidence, not new choosing results. Distances below are the straight-line distance between the guess and the true camera error, in millimetres. The paired start column uses only the same episodes that subsequently succeeded.

| Practice seed | Successes / 500 | Start, all | First contact | At success | Start, successful only | End at step 200 | End closer than start | First contact step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 327 | 0.953 | 1.153 | 1.654 | 0.761 | 2.198 | 11.2% | 127.8 |
| 101 | 322 | 0.953 | 1.556 | 1.302 | 0.789 | 2.413 | 16.2% | 126.9 |
| 102 | 304 | 0.953 | 1.013 | 1.153 | 0.738 | 1.765 | 20.2% | 134.5 |
| 103 | 286 | 0.953 | 1.002 | 1.074 | 0.730 | 1.654 | 24.2% | 136.3 |
| 104 | 330 | 0.953 | 1.169 | 2.213 | 0.786 | 1.809 | 18.4% | 130.7 |

The outside check was mostly right: contact errors were 1.00 to 1.56 mm and end errors were 1.65 to 2.41 mm, against a mean true error of 0.953 mm. Only 11.2 to 24.2 percent ended closer, and contact occurred around steps 127 to 136. Each seed tended toward a different direction, but “whatever the true error” is too strong: the per-axis correlations between final guess and true error ranged from -0.37 to +0.52. Even at insertion, the guess was farther from the truth than its paired starting guess in every seed; continued motion after success exaggerates the end error but does not explain away the failure to find the camera error.

A fixed linear predictor, fitted only on each run's own saved 200 practice episodes, had mean first-contact errors of about 0.919, 0.945 and 0.956 mm for windows of 1, 20 and 40 readings; a zero guess was about 0.951 mm. These tests do not support a benefit from a longer linear history, but do not rule out a nonlinear learner using memory. The predictor was fixed before evaluation, and no test episode supplied a fitted parameter. The readings before control step 125 were barely predictive with any window. The full per-seed results, including prediction fit details, are in [the diagnostic data](results/takehome/followup/A/diagnosis.json) and [prediction table](results/takehome/followup/A/prediction_table.csv).

The saved original-policy diagnostic repeat succeeded in 86/99 episodes (86.9 percent) below 0.5 mm of bias, versus 8/67 (11.9 percent) above 1.5 mm. Those small groups support the importance of the camera error. This particular saved repeat totals 303/500 and does not replace the original 304/500 confirmation floor.

![Phase A diagnostics](results/takehome/followup/A/diagnosis.png)
