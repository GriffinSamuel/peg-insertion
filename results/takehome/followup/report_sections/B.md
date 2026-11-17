## Constant-offset sweep and revised ceiling rule

The prescribed 7 by 7 sweep covered guesses from -0.9 to +0.9 mm in 0.3 mm steps, with zero camera error and 256 test-only episodes per cell at seed 123. The best score was **242/256 (94.53 percent)**, tied at **(0, 0) mm and (+0.3, 0) mm**. A positive guess is subtracted from the camera reading, so the second cell shifts that reading by (-0.3, 0) mm. No tested constant offset beat the centre. The sweep therefore does not establish a higher constant-offset ceiling; it is a finite-grid measurement, not a proven upper bound. All 12,544 tests had no wall-passing or invalid-number flags, and none was learned from.

![Constant-offset sweep](results/takehome/followup/B/offset_sweep/heatmap.png)

Griffin revised the rule because the old policy's directional preference can allow a consistent offset to beat its zero-error reference. Above-reference aim results in fixed, one-sided and mixed conditions now receive an extra test and physical review instead of a halt. The normal-condition aim-only tripwire remains unchanged. The revised plans and reason were committed before continuation as b46779e. The observed sweep tie is retained even though it does not establish the proposed explanation for the earlier high result.

## Phase B: what helped and what did not

No isolated normal-condition change met the working-aim definition. Labels from successful practice improved paired success-time guess error by 0.308 mm, but reduced insertion to 47.07 percent versus 63.87 percent for matched old aim seeds: HURT. Constant-offset fill collapsed on one seed: UNSTABLE. Gate, pullback, memory, faster contact nudges, the normal heat map and within-episode search were NO REAL DIFFERENCE. Memory worsened paired guess error by 1.004 mm. The optional memory-plus-constant-fill rating map and general stack were not launched because their isolated prerequisites did not help; both remain scoreboard rows.

Plain aim on one-sided error qualified as **HELPED**, at 80.20 percent with 4.24 points of spread over five seeds, versus its 69.40 percent floor. This is a separate development result, not the original 80.2 percent confirmation headline. Its guess still did not find the true error: mean error at success was 1.310 mm versus 0.939 mm initially; the paired comparison worsened by 0.465 mm. The one-sided heat map scored 78.73 percent on three seeds, within its matched base's spread, so it was not extended. Its start/success errors were 1.715/1.054 mm.

Plain fixed-camera aim finished at 90.36 percent with 5.81 points of spread over five seeds; plain movement reached 93.84 percent with 6.76 points of spread. Their matched floor was 72.52 percent. Large variation between calibration seeds kept both verdicts at NO REAL DIFFERENCE under the predeclared conservative spread rule. The fixed heat map scored 92.40 percent on three seeds, also NO REAL DIFFERENCE against matched plain aim. High fixed-camera success does not establish that the guess identified the true error: plain fixed aim's start/success errors were 0.866/1.797 mm.

## Preserved halt and fresh restart

The original fixed-camera seed 4 attempt correctly stopped under the former rule after 160 charged practice episodes, at 256/256 curve-test successes against 470/500 for its reference. All 41 files in that directory remain unchanged. Its own DID NOT RUN row consumes 160 practice episodes, but is not an additional setting or a completed final seed. Success-time guess error is unavailable because its final 500-episode test did not run and full curve traces were deliberately not saved.

The authorized restart began from scratch in a separate directory with its own 200-episode ledger and finished at 492/500. The five-seed result includes that completed restart, not the partial curve score. The [original halt record](results/takehome/followup/B/tripwire.json) and [fresh result](results/takehome/followup/B/B_aim_fixed/seed_4_restart_1/result.json) preserve both outcomes.
