# Take-home progress

## CURRENT STATE (2026-11-17, takeover finished)
- ALL PHASES DONE. Report: TAKEHOME_FOLLOWUP_REPORT.md. Fresh seeds 200-204, test 1001: plain movement 84.2% +- 5.9 (floor 61.4, zero-error 95.0). Tests 143 passed, original files verified.
- PARKED: torque-direction and four-sided touch probes (not implemented), mechanism replays (not run), fresh row for E_normal_move_then_aim (locked procedure runs chosen method only). B_aim_fixed_seed4_halted kept as its own row.
- Advisor consultations used: 3 (aim mechanism, touch go/no-go, final choice review), all in results/takehome/followup/advisor/.

## Current status: follow-up Phase B complete; mixed comparisons next (2026-11-16)

The revised consistent-camera rule is committed as b46779e. The offset sweep finished first and tied the centre at94.53%. All interrupted B items and PROMISING extensions are complete; no promotions remain. Plain one-sided aim HELPED at80.20%±4.24 points; no normal-camera aim fix met the working definition. The original160-episode seed4 halt is a preserved row; its fresh200-episode restart scored492/500. Commit/push B, then run mixed comparisons before C to G. Fresh seeds remain unused; the original80.2% result is unchanged.

## Original-study completion (2026-11-15)

All70 original/new tests passed. Fresh final-method practice seed100/test456 scored378/500 (75.6%), within the6.52-point sample spread around the80.20% confirmation mean; its own200-episode budget and frozen-reference audit passed. Original code, tests, policies and results are unchanged outside results/takehome. Validation artifacts were committed/pushed and local/remote commit equality verified; final handoff is saved on takehome. Final totals:29010 practice episodes in146 ledgers,409690 full test-only episodes,191 reset-only probes,28 searched settings. Full traces, replay, checkpoints and videos remain local and ignored. No experiments remain running. Exact trajectory repeatability remains the measured limitation described below.

Read TAKEHOME_REPORT.md first, then results/takehome/repeatability_findings.md and the dated phase reports. Final fallback and open-question lists are at the end of this log. No further tuning or result replacement was performed.

## Historical status after revised instructions, 2026-11-15

Re-read AGENTS.md, CODEX_GOALS.md and TAKEHOME_PLAN.md in full. Griffin resolved the sampling question: exactly half fill-up and half learner data when both exist; otherwise use the nonempty source. The historical pause below is resolved. Goals 1–3 remain passed under the revised checks. Goal 4 now builds only the main learner, its configurable output dimension and the explicitly listed basic settings, replay metadata and test traces. Later tricks are deferred until their respective goals.

Corrections to existing work: added local .gitignore files to exclude generated caches and test scratch files; superseded the old broad Goal 4 checklist with goal4_requirements_revised.json; updated the logs to use the single stopping rule. No existing baseline, mechanism result, seed or test needs changing. Commit/push the current work before resuming Goal 4. Heading 8 in historical reports is retained as history, not an active request.

## 2026-11-15: Preflight complete; Goal 1 in progress

All ten requested preflight checks passed. Active lead: gpt-6-astra, xhigh. Two concurrent gpt-6-luna/low helpers were verified from their runtime records. Branch takehome; WSL2; project virtualenv Python; JAX [CudaDevice(id=0)]. Temporary create/read/delete checks passed in both permitted directories. A detached GPU check survived its launching shell and completed a 30-second sleep. The eight test-only episodes succeeded 2/8 in 63.77 seconds including imports/compilation (31.71 seconds for rollout), with finite statistics and no through-wall episodes. No learning occurred. Old preflight outputs and cache were removed before the new checks.

Evidence: results/takehome/preflight_rollout.json, preflight_detached.log, preflight_helper_readme.json, preflight_helper_tests.json, and preflight_verification.json. Tracked files were unchanged at the end of preflight. The helper specification lives in takehome/routine_helper.toml and is applied through explicit spawn options, not automatic .codex discovery.

Limits: four concurrent agents including the lead, reported context window 258400 tokens, account rate limits unavailable, no visible fixed session deadline. GPU 8 GiB; available host memory about 16 GiB; free disk about 336 GB at inspection. Detached jobs cannot survive a WSL shutdown. No uninterrupted-run guarantee is possible.

Goal 1 command: source scripts/env.sh && python -B takehome/baseline.py. The run evaluates 500 episodes per condition, test seed 123, in one process. Goal checks and a complete phase report will follow its results.

Practice episodes used: 0. Learning settings tried: 0. Resume from Goal 1 until its report records all checks passing.

## 2026-11-15: Goal 1 complete

### 1. What I did (short list)
- Ran takehome/baseline.py detached; both conditions shared one Python process.
- Reused the existing loader, rollout, deterministic policy, physics randomization, and success rule.
- Saved aggregate and per-episode results under results/takehome/baseline*.

### 2. What I found (the numbers, with how many test episodes and seeds)
- Normal: 307/500, 61.4%; 95% interval 57.1–65.6%.
- No constant camera bias: 472/500, 94.4%; 95% interval 92.0–96.1%.
- One fixed test seed, 123; neither test was learned from. Rollouts took 23.78 and 8.95 seconds including their compilation. Script time excluding imports: 41.09 seconds.

### 3. Checks: each check from the goal, marked pass or fail
- PASS: normal within 5 percentage points of 62.8% (difference -1.4 points).
- PASS: no-bias result within 5 percentage points of 94.0% (difference +0.4 points).
- PASS: GPU visible to JAX before evaluation; preflight also confirmed it before project work.
- Additional PASS: all stored statistics finite; zero through-wall episodes in both conditions.

### 4. Surprises or problems
Results differ slightly from the historical runs, within the predeclared tolerance; the cause has not been established. The eight-episode preflight was 25%, illustrating why that small sample is not the baseline.

### 5. Decisions I logged
Preserve the existing evaluator's batch/contact settings; remove only constant camera bias for the historical ceiling; retain per-step observation noise; redirect and reuse caches.

### 6. Practice episodes used, and settings tried so far
0 practice episodes; 0 learning settings. Test-only episodes: 1008 (8 preflight plus 1000 baseline).

### 7. Plain-language explanation for Griffin (five sentences at most)
The original policy succeeds about six times in ten. Removing its constant camera error raises that to about nine times in ten. This supports testing whether a learned correction can recover some of that gap. The ceiling still includes small random camera jitter. No learning has happened yet.

### 8. Questions before the next phase
None. Continue with Goal 2.

## 2026-11-15: Goal 2 complete

### 1. What I did (short list)
- Built takehome/correction.py with a two-value capped running estimate and the frozen policy acting on corrected observations.
- Added first-success reward and distinct learner termination/time-limit flags; original environment code remains unchanged.
- Ran three tests with the existing GPU simulator; results/takehome/goal2_tests.exit.json records exit code 0.

### 2. What I found (the numbers, with how many test episodes and seeds)
Three tests passed in 38.81 seconds (45.08 seconds including runner overhead). Four complete test-only episodes at seed 123 checked policy immutability; two four-environment reset probes checked zero initialization. No learning occurred.

### 3. Checks: each check from the goal, marked pass or fail
- PASS: guess resets to zero and saturates at +/-3 mm per axis.
- PASS: reward is 1 only at first insertion; success terminates the learner and the time limit sets a separate truncation flag.
- PASS: actual in-memory policy parameter hash and checkpoint bytes were unchanged after the test rollout.

### 4. Surprises or problems
An existing dependency emits a warning that JAXopt is no longer maintained; tests still pass. No fallback used.

### 5. Decisions I logged
0.1 mm step cap, 3 mm estimate cap; subclass wrapper to remain compatible with existing domain randomization; continue physics for original fixed-length test measurements while masking ended learner transitions.

### 6. Practice episodes used, and settings tried so far
0 practice episodes, 0 learning settings; 1012 complete test-only episodes cumulatively, plus eight reset-only probes.

### 7. Plain-language explanation for Griffin (five sentences at most)
The wrapper now adjusts the camera reading before the original policy acts. Its estimate starts at zero and cannot grow beyond the chosen bound. It gives one success reward and distinguishes running out of time from an actual terminal event. The original policy's weights did not change. Next we test whether this machinery preserves and then improves its behavior when given the true error.

### 8. Questions before the next phase
None. Continue with Goal 3 using takehome/mechanism.py.

## 2026-11-15: Goal 3 complete

### 1. What I did (short list)
- Ran zero correction and a test-only oracle that supplies the true constant bias.
- Used the existing rollout for 500 episodes per condition at test seed 123.
- Saved results in results/takehome/mechanism_check.json and condition-level JSON/NPZ files.

### 2. What I found (the numbers, with how many test episodes and seeds)
Zero: 302/500 (60.4%), 1.0 percentage point below the Goal 1 normal result. Oracle: 471/500 (94.2%), 0.2 points below the no-bias ceiling. One test seed, 123; 1000 episodes, none learned from. Rollouts took 14.05 and 9.11 seconds; full detached job took 47.39 seconds and exited 0.

### 3. Checks: each check from the goal, marked pass or fail
- PASS: zero is within 3 points of the Goal 1 normal result on the same seed.
- PASS: oracle is within 5 points of the no-bias ceiling.
- Additional PASS: frozen in-memory parameters unchanged, all statistics finite, no detected through-wall episodes.

### 4. Surprises or problems
The zero run does not reproduce every individual baseline success bit. The difference is within the tolerance specified before testing; its cause has not been established. Do not describe the implementation as bitwise-equivalent. No fallback used.

### 5. Decisions I logged
No new settings selected from these results. Keep aim correction as the main method, as the plan prescribes after a passing oracle check.

### 6. Practice episodes used, and settings tried so far
0 practice episodes; 0 learning settings. Cumulative complete test-only episodes: 2012, plus eight reset-only probes.

### 7. Plain-language explanation for Griffin (five sentences at most)
The correction mechanism preserves the original success rate within the agreed tolerance. When given the true error, it recovers almost all the available improvement. This establishes that correcting the observation can help. It does not establish that a learner can infer the error within 200 episodes. That is the next test.

### 8. Questions before the next phase
None from the mechanism checks. Continue to Goal 4.

## 2026-11-15: Goal 4 incomplete — clarification pending under AGENTS.md rule 8

### 1. What I did (short list)
- Added takehome/budget.py: locked, persistent practice reservations, with a hard maximum of 200 and a guard rejecting test-labelled/unreserved data.
- Ran takehome/test_budget.py: two tests passed in 0.29 seconds; these use synthetic ledger entries, not simulator episodes.
- Collected exact Goal 4/7 switch requirements in results/takehome/goal4_switch_requirements.json.
- Asked Griffin to resolve the sampling-rule conflict before implementing dependent learner behavior.

### 2. What I found (the numbers, with how many test episodes and seeds)
The guard rejects reservation 201 after 40 fill-up and 160 online synthetic reservations and still rejects it after reloading the ledger. It also rejects test-labelled data, unknown IDs, and mismatched data sources. No learning trial or additional simulation was run in Goal 4.

### 3. Checks: each check from the goal, marked pass or fail
- FAIL / not yet run: 10-episode end-to-end learning trial.
- PASS: counter refuses episode 201 and persists the reservation count.
- FAIL / not yet implemented: one command starts a full 200-episode learning experiment.
- FAIL / incomplete: replay sampling, correction rescaling, and checkpoint input normalization are not yet implemented in a learner.
- FAIL / pending clarification and implementation: all Goal 7 switches.

### 4. Surprises or problems
CODEX_GOALS.md line 75 requires every batch to contain half fill-up and half new data; line 76 requires all Goal 7 switches. Lines 99 and 101 request zero fill-up and a scorekeeper briefed first, respectively. A source containing zero records cannot supply half a batch. This is a requirements conflict, not a failed experiment or preflight check. AGENTS.md rule 8 says to ask rather than quietly work around an impossible plan requirement.

### 5. Decisions I logged
Persist reservations before collection and count interrupted reservations conservatively. No sampling exception has been assumed or implemented.

### 6. Practice episodes used, and settings tried so far
0 practice episodes, 0 learning settings; 2012 complete test-only episodes plus eight reset-only probes cumulatively. Budget unit tests contain synthetic entries and collect no episodes.

### 7. Plain-language explanation for Griffin (five sentences at most)
The first three goals passed. A budget guard now prevents an experiment from collecting episode 201. The next requirement says to mix two kinds of experience equally, but also requires experiments where one kind does not exist. I have asked which rule should apply in those cases. No training or experimental setting selection has happened.

### 8. Questions before the next phase
May the required zero-fill-up and critic-pretraining cases sample only from the available source, while using exactly 50/50 whenever both sources exist? Alternatively, should strict 50/50 remain mandatory and those incompatible experiments be removed? Resume Goal 4 after Griffin answers; do not skip to Goal 5.

## 2026-11-15: Goal 4 complete under the revised requirements

### 1. What I did (short list)
- Built the main deterministic actor-critic, a layer-normalized critic ensemble, exact symmetric replay sampling with the approved single-source exceptions, and a persistent budget guard.
- Added collection in ten-episode batches, complete replay provenance, a separate test trace runner, checkpoint saving, and resume support after completed batches.
- Reused checkpoint normalization and the existing rollout's environment wrapping, randomization, seed partition, summary and physical-validity checks.
- Ran 11 contract tests and one ten-episode code trial in one process; reviewed the helper's standalone Goal 5 plotting script before use.

### 2. What I found (the numbers, with how many test episodes and seeds)
11 tests passed in 49.97 seconds. The code trial used practice seed 999, exactly 10 learner episodes, zero fill-up, 1837 valid transitions and 7348 critic updates. Its separate eight-episode test at seed 123 scored 0/8, with finite simulation/model values and no through-wall episodes. Trial execution took 43.68 seconds in the already-started process; all checks and the trial took 116.29 seconds including process startup. This is a code test, not an experimental setting, and its poor result is retained without tuning the main configuration.

### 3. Checks: each check from the goal, marked pass or fail
- PASS: ten-episode end-to-end run finished with exit code 0 and no invalid simulation/model numbers.
- PASS: budget guard rejects episode 201, persists reservations, and rejects test-labelled or unreserved data.
- PASS: source scripts/env.sh && python -B takehome/train.py --policy force_0.5mm_s2 --seeds 0 starts the full 200-episode experiment.
- PASS: compiled sampler returns exactly 64 fill-up and 64 learner transitions when both exist; all 128 come from the available source otherwise.
- PASS: critic receives normalized physical corrections; original sensor scaling exactly matches the frozen policy's checkpoint statistics.
- PASS: fill-up count, updates per step, ensemble size and aim caps are command-line settings; 2/3/5-output networks all initialize to exactly zero. Movement environment adapters and other later tricks are deferred to their prescribed goals.
- PASS: ten-episode trial is marked code_test=true and setting_counted=false.
- PASS: replay records success/timeout separately, episode and step IDs, source, old and executed actions, and true error; true error is excluded from training inputs.
- PASS: separate test traces contain running guess, true error, Q rating and force at every step, with explicit test_only=true and learned_from=false labels.

### 4. Surprises or problems
The tiny code-test learner scored 0/8. No success target applies to this check, and this observation was not used to select different hyperparameters. Mean insertion time for zero successes is undefined and is serialized as null, not treated as an invalid simulation value. No fix attempt or fallback was required. The helper assigned to launch checks had not created the log, so the lead launched the reviewed command and retained the helper for monitoring.

### 5. Decisions I logged
PLAN GAP entries fix architecture, update ratio, learning rate, discount and target averaging, exploration noise, scaling, code-trial seed/split/test size, fill-up-end curve point, deferred adapters, and the nine-run timing schedule before full experiments. Raw Q is explicitly not called a success probability. Four timing repeats supplement the five designated main runs; each has an independent 200-episode ledger and cannot replace a headline run based on its score.

### 6. Practice episodes used, and settings tried so far
10 practice episodes in the code trial; 0 reported learning settings tried. Cumulative full test-only episodes: 2024, plus 16 reset-only probes. Goal 5 will add 1800 practice episodes across nine separate 200-episode experiments of one setting, including four timing repeats.

### 7. Plain-language explanation for Griffin (five sentences at most)
The learner can now collect practice, reuse that experience, update its networks, and save the result. The counter and data checks keep tests out of learning and prevent a run from exceeding its budget. The small code trial worked mechanically but performed poorly. That does not predict the outcome of the full 40-fill-up/160-learner experiment. We will report every full run with the settings fixed now.

### 8. Questions before the next phase
No BLOCKING issue. Commit and push Goal 4, then run Goal 5 and stop at its check-in.

## 2026-11-15: Goal 5 complete, planned check-in

### 1. What I did (short list)
- Completed the predeclared detached nine-run benchmark: five designated main seeds and four timing repeats, each with its own 200-episode ledger.
- Ran the main aim-only setting with 40 fill-up and 160 learner episodes, saving every learning curve, final model, replay and test trace.
- Reviewed helper plots, all-run tables and data/source audits; corrected overlapping plot headings without changing any result.
- Wrote TAKEHOME_REPORT.md with the measured results, references, runtime comparison, per-goal time estimates, cut list and visible usage limits.

### 2. What I found (the numbers, with how many test episodes and seeds)
Final main results for practice seeds 0, 1, 2, 3 and 4: 365/500 (73.0%), 278/500 (55.6%), 315/500 (63.0%), 305/500 (61.0%) and 366/500 (73.2%). All tests use seed 123. Mean 65.16%; sample standard deviation 7.74 percentage points. The mean is 3.76 points above the original-policy floor of 61.4%, but 29.24 points below the 94.4% no-camera-bias ceiling. The gain is smaller than the spread and does not establish reliable improvement. Goal 3 remains passed: zero correction 60.4%, supplied true bias 94.2%, each on 500 tests at seed 123.

Each main curve has point 0 from the original 500-test baseline, eight intermediate 256-test points at 40 through 180, and the final 500-test point at 200. The four timing repeats scored 318/500 (seed 0, four processes), 282/500 (seed 1, four processes), 314/500 and 312/500 (seeds 0 and 1 sequentially in one process). All nine results are in results/takehome/goal5_all_runs.csv; none is discarded.

Measured wall times including startup: one run alone 116.03 seconds; two processes, two runs 132.05 seconds; four processes, four runs 232.04 seconds; two seeds in one process 156.03 seconds. Effective time per run is respectively 116.03, 66.02, 58.01 and 78.01 seconds. Full detached benchmark: 637.02 seconds, exit 0. The fastest observed setup is four processes. Remaining-work estimate: 13.6 hours serial, 8.1 hours with that setup and helpers, conservative worst-case allowance 32.5 hours; full per-goal table and assumptions are in results/takehome/time_estimate.md and TAKEHOME_REPORT.md.

### 3. Checks: each check from the goal, marked pass or fail
- PASS: all five practice seeds finished, and each final result and complete curve is shown.
- PASS: final result is compared with the measured Goal 1 floor and ceiling.
- PASS: report plainly states a small measured average improvement with mixed runs, insufficient to establish reliable improvement; no target score was imposed.
- PASS: real 200-episode runs compared one, two and four GPU processes and multiple seeds in one process; all repeats are retained and identified.
- PASS: check-in includes Goal 3 evidence, curves, measured timing, per-goal run counts and runtime estimates, likely/worst totals, ordered cuts and allowance visibility.
- Additional PASS: every budget is exactly 200 with 40 fill-up/160 learner entries, tests remain separate, model/loss/final trace values are finite, frozen policy and seven core source hashes are unchanged, and final through-wall counts are zero. The five learned per-episode outcome arrays are not identical. No confirmation seeds were used. No original source/policies/results outside results/takehome were changed.

### 4. Surprises or problems
Several curves deteriorate sharply after learning starts, then recover. Same-seed repeats differ: seed 0 ranges from 62.8% to 73.0%. In the saved first fill-up batches, biases, proposed random actions and running guesses match exactly, but sensor readings and contact forces already differ before learning. The cause is unresolved, so neither bitwise repeatability nor a robust benefit is claimed. Core source hashes match across all runs. No tripwire or experimental fallback was triggered. One report-assembly lookup used the wrong JSON field name for through-wall counts; the field was corrected to through_wall_episodes and the read-only check passed without rerunning or changing any experiment.

### 5. Decisions I logged
Use four concurrent GPU processes provisionally for future batches because measured throughput is highest. Keep the predesignated main results despite differing timing repeats. Record explicit PLAN GAP forecasting assumptions for unspecified future setting counts, video counts, code overhead and runtime multipliers. No proposed cut is applied. Settings, success rules, seeds and tests were not changed after observing scores.

### 6. Practice episodes used, and settings tried so far
1810 practice episodes total: 1000 across the five designated main experiments, 800 across four independent timing repeats, and 10 in the earlier code trial. Each full experiment remained at exactly 200; the trial remained at 10. One distinct learning setting tried. Cumulative complete test-only episodes: 24,956, plus 16 reset-only probes. Test episodes were never learned from. Settings for Goals 6 and 7 and confirmation seeds remain unused.

### 7. Plain-language explanation for Griffin (five sentences at most)
Correcting the camera reading helped some runs and hurt others. The average rose from 61.4% to 65.16%, but the differences between runs were larger than that gain. Giving the system the true error still reaches about 94%, so the main learner has not recovered most of the available benefit. Four GPU processes finished the most experiments per minute in this benchmark. Repeated seeds also varied, and that remains a limitation rather than a reason to replace inconvenient results.

### 8. Questions before the next phase
BLOCKING: planned Goal 5 check-in under CODEX_GOALS.md's When to stop rule, not a failed goal. Read TAKEHOME_REPORT.md and decide whether to continue as written or apply cuts. The likely remaining laptop-awake estimate is 8.1 hours with the fastest measured setup and helpers; the conservative worst-case allowance is 32.5 hours. Account usage allowance is unavailable (rate_limits is null); no percentage used can be inferred from cumulative token counters. Commit and push this completed goal, then wait for Griffin; do not begin Goal 6.

## 2026-11-15: Goal 6 complete, equal-budget comparisons

### 1. What I did (short list)
- Added bounded movement-only and joint aim/movement controls, with correctly scaled critic inputs and exactly zero initial outputs.
- Added a clearly labelled supervised reference that cheats on information by learning true camera-bias labels from its own counted practice episodes.
- Completed 15 new full experiments, reused the five designated Goal 5 aim results, and reviewed separate helper charts and audits.
- Prepared the Goal 7 design and isolated feature code during detached runs; 14 synthetic feature contracts passed, with integration checks still required before screening.

### 2. What I found (the numbers, with how many test episodes and seeds)
| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | Sample spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aim only | 365/500 | 278/500 | 315/500 | 305/500 | 366/500 | 65.16% | 7.74 points |
| Movement only | 425/500 | 411/500 | 427/500 | 428/500 | 393/500 | 83.36% | 2.99 points |
| Aim and movement | 421/500 | 378/500 | 363/500 | 407/500 | 396/500 | 78.60% | 4.60 points |
| Supervised reference (cheats: true labels) | 325/500 | 349/500 | 328/500 | 327/500 | 325/500 | 66.16% | 2.05 points |

Every row uses practice seeds 0 to 4 and a separate final 500-episode test at seed 123. Original floor 61.4%; no-camera-bias ceiling 94.4%. Movement-only did best among the three eligible learners: +21.96 percentage points over the floor and +18.20 over aim-only. Adding movement to aim increased the mean by 13.44 points; adding aim to movement decreased it by 4.76 points under this budget and recipe. The supervised reference (cheats: true labels) averaged 66.16%; learning bias from ordinary observations remains difficult even with labelled targets, and this trained reference is not an upper bound on what any supervised learner could achieve. All 15 new runs finished in 693.38 seconds including process startup, about 11.6 minutes.

### 3. Checks: each check from the goal, marked pass or fail
- PASS: all four learned methods include practice seeds 0 to 4, final 500 tests at seed 123, and exactly 200 own practice episodes per run; floor and ceiling are reused no-learning references.
- PASS: combined network outputs start exactly zero; zero movement leaves the old action unchanged, and zero aim leaves its reading unchanged. The 15 Goal 6 contract checks passed.
- PASS: the supervised reference (cheats: true labels) is explicitly labelled in its configuration/results, comparison table and plots; it uses the same budget counter.
- PASS: the report identifies movement-only as best and states what adding each component changed.
- PASS: the artifact audit reports zero failures across all20 learned results, including finite arrays, frozen policies, exact ledgers, separate tests, distinct seed outcome arrays, no through-wall episodes and the final-trace-only storage rule for new runs. Charts were reviewed; shared legends sit outside curves to keep failures visible.

### 4. Surprises or problems
The joint learner performed worse than movement alone. True-label supervision did not approach the true-error oracle; access to labels during training is not the same as being given the actual error while acting. No ceiling exception, invalid numerical result, budget overrun, or learned test episode occurred. A shared JAX cache lookup warning was nonfatal; all six worker jobs exited zero. The previously diagnosed simulator nondeterminism remains a limitation and results were not replaced.

### 5. Decisions I logged
Movement correction cap is 0.2 mm per axis per step with original actuator clipping; supervised labels train a direct bias estimate converted to bounded aim nudges. The manifest fixes all new seeds and preserves the five old designated runs. A seed-789 extra test is predeclared for movement-capable ceiling exceedances but was not needed. Goal 7 numeric choices, compatibility restrictions and conservative comparison rules were written before screening; all named tricks remain included.

### 6. Practice episodes used, and settings tried so far
3000 new practice episodes across 15 independent 200-episode experiments; cumulative practice 4810. Four distinct settings tried, including the supervised cheating reference; no Goal 7 setting has yet collected practice. Goal 6 added 38,220 experimental test episodes and four contract-test episodes, for 64,830 complete test-only episodes cumulatively, plus 174 reset-only probes. The 14 prepared Goal 7 unit tests used synthetic data and collected no episodes. Full final traces, practice replay and model checkpoints remain local and ignored by git.

### 7. Plain-language explanation for Griffin (five sentences at most)
Small corrections to the robot's movement worked better than corrections to its camera reading in these runs. Combining both helped compared with aim alone but did worse than movement alone. The learner trained with true-error labels also struggled to infer the error from what it could observe. These are development results, so they are not the final headline chosen after tuning. The next goal tests every planned variation, then checks the chosen method on untouched confirmation seeds.

### 8. Questions before the next phase
No blocking issue or new question. Commit and push Goal 6, finish Goal 7 integration checks, and continue through Goal 13 under the approved no-cuts instruction.

## 2026-11-15: Goal 7 started

Goal 6 was committed and pushed as 2a0523d. All 32 integration/regression contracts passed in 59.49 seconds before screening. The predeclared 24-setting/72-run screening suite started detached; every experiment owns its budget, and every setting checks the identical-outcome tripwire as soon as all its seeds finish. No confirmation seed has been used. Intermediate curves retain seed123/256 tests; the later confirmation final500 will use seed456, as logged. Test-only counts before screening are 64,842 full episodes and 182 reset-only probes; practice remains 4810 before new screening reservations. The lead is preparing selection/reporting code while GPU jobs run, and helpers are preparing independent plots and audits.


## 2026-11-15: Goal 7 complete

### 1. What I did (short list)

- Finished all24 new screening settings on72 independent runs; audited every run and all87 new screening/confirmation core-source hashes.
- Applied the fixed selection rule before confirmation. No eligible single trick clearly improved its relevant base, so round2 had no combinations.
- Confirmed plain aim, movement and combined on all five fresh practice seeds; visually reviewed both confirmation charts.

### 2. What I found (the numbers, with how many test episodes and seeds)

Final500-test counts at test seed456, practice seeds100-104: aim327,322,304,286,330 (62.76% ±3.71pp sample SD); movement380,402,420,360,443 (80.20% ±6.52pp); combined289,422,274,381,326 (67.68% ±12.47pp). The selected movement headline is lower than development83.36%, and remains the headline. References304/500=60.8% and463/500=92.6%. Screening took3230.14 seconds; confirmation670.14 seconds, plus41.74 seconds for confirmation references.

### 3. Checks: each check from the goal, marked pass or fail

- PASS: selection used development seeds0-4 (screening0-2), final500 at123; choice timestamp precedes all confirmation work.
- PASS: every experiment had its own exact200-episode ledger, including fill-up.
- PASS:28 settings stated, below45 cap; all settings and seeds shown.
- PASS: differences smaller than the larger matched seed spread are labelled unproven, not real gains.
- PASS: fixed selection rule independently reapplied and agrees with the immutable written movement choice.
- PASS: chosen method confirmed at100-104/456 and80.20% reported as headline despite lower score.
- PASS: all three plain comparisons confirmed, fresh references supplied, and28 versus1 versus1 selection advantage stated.

### 4. Surprises or problems

Aim and combined show severe early learning-curve drops, retained without filtering. Combined varied widely on fresh seeds. No tripwire fired; all final physical checks reported zero through-wall episodes. The28-setting selection advantage matters even though the winner has no added trick. Repeatability remains limited.

### 5. Decisions I logged

The immutable choice, absence of qualifying combinations, sequential-noise interpretation, and required ceiling checks for later diagnostic repeats are logged. Preparatory camera, position-input, calibration, failure-partition and ceiling-helper checks passed; later real phases remain gated. All PLAN GAP entries remain in the decision log.

### 6. Practice episodes used, and settings tried so far

Goal7 added17,400 practice episodes (87 runs×200). Cumulative22,210 practice episodes in112 complete ledgers,28 settings,285,226 full test-only episodes and182 reset-only probes. The global total includes the separate code trial and timing repeats; the budget is200 per experiment, never200 for the whole study. Small result files only are committed; binary traces/models/videos stay local.

### 7. Plain-language explanation for Griffin (five sentences at most)

Correcting hand movements was the strongest approach. On the fresh test it succeeded about80 times out of100, versus61 for the old policy. Fixing the camera estimate alone did not show a clear improvement. Trying many settings helped choose the winner, and that advantage is disclosed. The next phases explain its confidence and remaining failures without changing it.

### 8. Questions before the next phase

None requiring a pause. Continue to Goal8 under the approved no-cuts instruction.


## 2026-11-15: Goal 8 complete

### 1. What I did (short list)

- Fitted45 separate episode-success predictors from the existing counted practice, across five seeds and nine checkpoints.
- Reused the final confirmation traces, collected only the intermediate traces needed for this analysis, and plotted probability reliability, raw ratings and training history.
- Listed every confident failed moment and every failed-episode confidence drop, keeping null where no drop met the rule.

### 2. What I found (the numbers, with how many test episodes and seeds)

Final pooled prediction69.54%, observed74.77% across392,329 valid steps from2,500 tests at seed456, practice seeds100-104. Signed error−5.23 percentage points: underconfident on average. Ten-bin expected calibration error6.32pp; Brier score0.15235. There were495 failed episodes,355 with a predicted chance≥80% at some moment, and15 with a≥25pp fall between adjacent five-step averages. Analysis took90.83 seconds; all45 fits converged.

### 3. Checks: each check from the goal, marked pass or fail

- PASS: charts use separate test episodes only, with per-seed and pooled results; audit has zero failures.
- PASS: report states the predictor is underconfident on average and distinguishes raw Q from episode probability.

### 4. Surprises or problems

Average underconfidence coexists with confident failed moments. Most failed episodes have no large confidence drop. Step-weighting makes74.77% different from the80.20% episode headline; these are not interchangeable. Raw Q versus outcome is not monotonic after pooling times and states.

### 5. Decisions I logged

Used the predeclared small predictor with remaining time, ten bins, fixed thresholds and own-practice-only fitting. No tuning or new practice. The intermediate full traces are the explicit Goal8 exception to final-only trace storage, and stay local. Installed and tested the passive Goal9 pose hook only after Goal7 training finished; removed temporary fixtures from git while keeping them local.

### 6. Practice episodes used, and settings tried so far

Zero new practice,22,210 cumulative practice,28 settings. Goal8 added10,240 intermediate test-only episodes;2,500 final tests were reused without double counting. The four pose integration tests bring the cumulative full test-only count to295,470, plus182 reset-only probes.

### 7. Plain-language explanation for Griffin (five sentences at most)

The learner's raw rating cannot be read as a percentage. A separate predictor trained on the same practice gives a more meaningful probability, but it still makes confident mistakes. Overall it predicts success less often than success actually occurs. It rarely gives a sharp warning before a failure. These plots describe its limits without changing the learner.

### 8. Questions before the next phase

None requiring a pause. Continue to failure analysis.


## 2026-11-15: Goal 9 complete

### 1. What I did (short list)

- Applied the predeclared ordered failure rules to all original confirmation traces and a separate500-test before repeat.
- Recorded labelled illustrative pose repeats and rendered13 videos through the existing rendering functions.
- Audited every video and reviewed all ending previews plus temporal samples; produced per-seed/pooled failure tables and chart.

### 2. What I found (the numbers, with how many test episodes and seeds)

Before repeat303/500 successes; failures: never-contact7, wrong-way0, rim22, residual timeout168, other0. After, across practice seeds100-104 and2,500 tests at456: successes2,005; failures1,216,68,210,0 in the same order. Each category's all-seed counts are in failures.json. Four separate pose repeats at practice seeds100-103 scored382,391,419,349/500; all remain labelled illustrative and do not replace scored tests. Pose analysis took54.03s; rendering142.39s.

### 3. Checks: each check from the goal, marked pass or fail

- PASS: every category is defined from recorded quantities with fixed thresholds.
- PASS: fixed order assigns each failure exactly once.
- PASS: other is included and counts sum to197 before and495 after failures.
- PASS: all13 videos reuse existing rendering code through takehome-only scripts; overlays, metadata,100 frames,25fps and4s duration checked.

### 4. Surprises or problems

The before traced repeat differs from the headline by one success, consistent with the known reproducibility limitation. Wrong-way labels do not prove causation and cannot occur without a correction; the remaining timeout category excludes earlier pattern matches. Near-contact is a joint5mm/3mm/5N proxy, not literal absence of approach. Only one after episode belonged to that rare category, so one example is shown; zero-case categories have no invented videos.

### 5. Decisions I logged

The fixed rule order, noisy outward-direction approximation, labelled pose-repeat policy, small preview artifacts and temporal review samples are all logged. No rules were changed after seeing counts.

### 6. Practice episodes used, and settings tried so far

Zero new practice,22,210 total,28 settings. Goal9 added2,500 test-only episodes (500 before plus four500-test pose repeats), bringing full test-only total to297,970;182 reset-only probes remain separate. Original2,500 final tests were reused without recounting.

### 7. Plain-language explanation for Griffin (five sentences at most)

The final method fails less often overall. The remaining failures include shallow contact, outward correction commands and failures to finish in time. These labels describe patterns, not proven causes. The videos show what those patterns look like with the true camera error visible. Repeated video runs are kept separate from the headline scores.

### 8. Questions before the next phase

None requiring a pause. Continue to the three-policy transfer check.


## 2026-11-15: Goal 10 complete

### 1. What I did (short list)

- Measured all three policies' own floor and no-bias reference before training.
- Reused the selected movement recipe unchanged for three independent practice seeds per policy.
- Audited all nine runs, read their counted practice outcomes, and plotted every seed against its own references.

### 2. What I found (the numbers, with how many test episodes and seeds)

All final tests: 500 episodes at seed 123, practice seeds 0, 1, 2. force_s4 scored 464, 442, 459 successes, mean 91.00% ±2.31pp sample SD, versus references 420/500 and 488/500. position_s1 scored 408, 403, 444, mean 83.67% ±4.47pp, versus 402/500 and 500/500. Hovering force_s1 scored 0, 0, 0, with both references zero and no successful episodes in any of its three 200-episode practice runs. References took 48.61 seconds; training suite 431.10 seconds.

### 3. Checks: each check from the goal, marked pass or fail

- PASS: all three policies reported, including the zero result, with a sentence explaining the evidence for each.
- PASS: references completed first; all nine algorithm configurations match confirmation except policy and evaluation context.
- PASS: position-only 10-input compatibility, frozen old weights, separate tests and each 200-episode ledger verified; full audit has zero failures.

### 4. Surprises or problems

The position-only policy already had an 80.4% floor, contrary to the expectation it would do poorly. Its observed gain is below its seed spread. The second force policy's seven-point gain is above its spread. The hovering checkpoint still cannot insert even without camera bias; with no practice success, this sparse-reward recipe did not discover a successful behavior. Its identical zero outcomes meet the explicit zero-floor-and-ceiling exemption.

### 5. Decisions I logged

No new algorithm choices or settings. Kept the final recipe unchanged and retained every result; policy-specific references and the required input-size handling were already specified. Plot bounds allow the zero and 100% markers to remain fully visible.

### 6. Practice episodes used, and settings tried so far

Added 1,800 practice episodes and 25,932 test-only episodes. Cumulative 24,010 practice episodes in 121 complete ledgers, 323,902 full test-only episodes and 182 reset-only probes. The search count remains 28; these are three new policy contexts, not tuned settings.

### 7. Plain-language explanation for Griffin (five sentences at most)

The method helped another force-sensing policy. The position-only policy already worked fairly well, and its small average gain is not clear beyond run-to-run variation. The hovering policy never found a successful insertion. Correcting a useful policy is easier here than teaching an unsuccessful one the whole task. These results limit the claim without hiding the failures.

### 8. Questions before the next phase

None requiring a pause. Continue to all eight camera conditions.


## 2026-11-15: Goal 11 complete

### 1. What I did (short list)

- Ran all eight predeclared camera conditions, with floor, true-error oracle, transfer and three fresh200-episode learners each.
- Audited all24 independent budgets and test provenance, collected practice outcomes and plotted all seeds.
- Reviewed four independent seed789 checks and three recorded trajectories from each.

### 2. What I found (the numbers, with how many test episodes and seeds)

Every final measurement has500 tests at seed123; learning seeds0,1,2. The no-added-error reference remains472/500. Suite runtime1127.46seconds.

- bias_2x: floor 153/500; transfer [244, 235, 248]; retrain [172, 239, 191]; retrain mean 40.13% ±6.91pp; oracle 470/500.
- bias_4x: floor 42/500; transfer [103, 77, 106]; retrain [68, 49, 88]; retrain mean 13.67% ±3.90pp; oracle 470/500.
- one_sided: floor 340/500; transfer [475, 385, 454]; retrain [446, 468, 450]; retrain mean 90.93% ±2.34pp; oracle 471/500.
- drift: floor 145/500; transfer [352, 270, 293]; retrain [359, 276, 357]; retrain mean 66.13% ±9.47pp; oracle 469/500.
- jitter: floor 339/500; transfer [424, 418, 431]; retrain [394, 358, 385]; retrain mean 75.80% ±3.75pp; oracle 470/500.
- rotation: floor 327/500; transfer [434, 418, 434]; retrain [449, 416, 422]; retrain mean 85.80% ±3.52pp; oracle 470/500.
- delay: floor 239/500; transfer [420, 379, 424]; retrain [405, 369, 239]; retrain mean 67.53% ±17.46pp; oracle 435/500.
- height: floor 373/500; transfer [465, 414, 414]; retrain [375, 410, 429]; retrain mean 80.93% ±5.48pp; oracle 488/500.

The four extra789 tests scored477,481,473,478/500, with zero wall or invalid-number flags. No extra score was used for selection.

### 3. Checks: each check from the goal, marked pass or fail

- PASS: every added error is implemented only in takehome code.
- PASS: all24 retrains use exactly their own200 episodes, with tests excluded from learning.
- PASS: height error is included, unchanged final recipe retained, all eight formulas and larger-error caps declared first.
- PASS: all scores, every seed, the table and chart are retained; full and context audits have zero remaining failures.

### 4. Surprises or problems

Four-times bias remains mostly unsuccessful despite a strong true-error oracle. Retraining is not consistently better than transfer, and delayed-reading retraining has large spread. Height error did not cause the expected failure. One audit assumed the generic display label; corrected once to match its declared baseline without changing any data or numeric check, and preserved its initial failure. The camera chart's legend used unlabelled patches; corrected to labelled bar containers and visually reviewed.

### 5. Decisions I logged

Logged the audit-label correction. The unchanged3D movement recipe for height, exact camera formulas, scaled aim caps and extra-seed physical checks were already declared. No new tuning choices.

### 6. Practice episodes used, and settings tried so far

Added4800 practice episodes and83152 test-only episodes, including2000 additional ceiling-verification tests. Totals:28810 practice episodes in145 completed ledgers,407054 full test-only episodes,182 reset-only probes;28 searched settings. Eight camera contexts are listed separately from tuning.

### 7. Plain-language explanation for Griffin (five sentences at most)

The method helps under several camera errors but does not solve them all. Very large offsets remain a clear limit. A one-sided offset is much easier. Retraining from scratch under a new error can be worse than keeping the original learner. The unexpected height result is reported as measured.

### 8. Questions before the next phase

None requiring a pause. Continue to the complete write-up and final validation.


## 2026-11-15: Goal 12 complete

### 1. What I did (short list)

- Rebuilt the complete report from audited summaries, including method/control flow, every seed, every setting, calibration, failure cases, policy and camera limits.
- Included all decisions and PLAN GAP entries, the budget assumption, AI assistance, reproducibility limitations and one-command figure rebuilds.
- Reviewed all prose claims and mechanically checked source hashes, table rows, links and figure commands.

### 2. What I found (the numbers, with how many test episodes and seeds)

Headline remains80.20% ±6.52 percentage points across confirmation practice seeds100-104, each tested on500 episodes at seed456, against60.8% original policy. All59 generated rows appear, all56 local links resolve, all14 figures have rebuild scripts, and35 result-source hashes match. All82 decisions and51 PLAN GAPs are present. No em dashes.

### 3. Checks: each check from the goal, marked pass or fail

- PASS: every reported measurement or parameter is traceable to small result files, protocol or decision inventory.
- PASS: lead review found no claim beyond the recorded evidence; causal explanations remain qualified, bad results retained and descriptive spread distinguished from a significance test.
- PASS: full mechanical report audit has zero failures; final Goal13 checks are explicitly pending rather than claimed early.

### 4. Surprises or problems

No new experimental issue. Updated stale figure-inventory wording after the final camera chart became available and clarified technical vocabulary. Historical log entries are retained as history, including superseded pauses and earlier cumulative counts.

### 5. Decisions I logged

No new algorithm or experimental choices. The report includes every existing decision, including the Goal11 audit-label repair and the storage policy.

### 6. Practice episodes used, and settings tried so far

No new episodes. Totals remain28810 practice episodes,407054 full test-only episodes,182 reset-only probes and28 settings.

### 7. Plain-language explanation for Griffin (five sentences at most)

The report tells the full story, including what did not work. Small movement corrections improved the starting policy more than the learned aim correction. Confidence estimates and transfer results have clear limits. The search advantage and all practice spent are visible. Final tests and one independent repeat are the only remaining work.

### 8. Questions before the next phase

None requiring a pause. Run Goal13 without further tuning.


## 2026-11-15: Goal 13 complete

### 1. What I did (short list)

- Ran all original and new tests unchanged in one detached command.
- Ran one fresh final-method experiment at the predeclared confirmation practice seed100/test456, starting in an empty directory with its own200-episode ledger.
- Audited the fresh run, protected git paths, ignored binaries, complete accounting and final report; committed and pushed validation artifacts and verified matching remote/local commits.

### 2. What I found (the numbers, with how many test episodes and seeds)

All70 tests passed in327.06seconds (331.98seconds including detached-process startup/shutdown), with one dependency deprecation warning. The fresh run took85.34seconds and scored378/500 (75.6%) at test456. Its4.60-point distance from the80.20% five-seed confirmation mean is below the6.52-point sample spread. Its original seed100 confirmation result was380/500 and remains unchanged. All result, budget and model hashes guarding that original confirmation match; the independent fresh audit has zero failures. Protected paths have an empty diff against main. No local-only binary artifact is tracked by git, and no eligible new file exceeds50MB.

### 3. Checks: each check from the goal, marked pass or fail

- PASS: all original and new tests passed.
- PASS: git diff main shows no changes to peg_insertion/, scripts/, tests/, original policies or original results outside results/takehome/.
- PASS: one fresh run is within one spread of the reported result, without changing its seed, test, budget or selected recipe.
- PASS: validation artifacts committed and pushed to origin takehome, with exact local/remote commit equality and clean status checked; final handoff record saved on that same branch.
- PASS: this final summary lists what to read, every continuation/availability fallback and open questions.

### 4. Surprises or problems

The fresh run is statistically consistent with the original confirmation spread but is not bit-for-bit identical. That does not erase the pre-learning simulator repeatability limitation. There was no experimental or unit-test failure in Goal13, no score replacement and no new fallback needed. The only test warning concerns an existing dependency's maintenance status.

### 5. Decisions I logged

No new experimental decisions. Used the seed100 rerun fixed before confirmation and retained its independent score. The original continuation decision and all82 entries/51 PLAN GAPs remain in the report.

### 6. Practice episodes used, and settings tried so far

Added200 practice episodes. Added2548 fresh-run test episodes (eight256-test curve points plus500 final) and88 full test episodes from the unit/integration suite, for2636 new full tests. Added9 reset-only probes; shorter physics probes are separately itemized. Final totals:29010 practice episodes in146 completed ledgers,409690 full test-only episodes,191 reset-only probes and28 searched settings. The three policy and eight camera contexts reused the final recipe without tuning.

### 7. Plain-language explanation for Griffin (five sentences at most)

The project is complete, and the final checks passed. Small movement corrections raised the confirmation average from60.8% to80.2%. One independent repeat landed inside the variation already reported. Large camera errors, an unsuccessful starting policy and exact repeatability remain limits. The report keeps those limits and all bad runs visible.

### 8. Questions before the next phase

No next automated phase or blocking question. Read the report and make sure you can explain it before submitting.

### Final handoff: what to read first

Read TAKEHOME_REPORT.md for the headline, method, all seeds, plots, limits, reproducibility commands and AI-assistance statement. Read results/takehome/repeatability_findings.md for the26m51s diagnosis and why production settings were left unchanged. This progress log and TAKEHOME_DECISIONS.md preserve the full history. Local videos are indexed by results/takehome/goal9/video_manifest.json; full videos, traces and model/replay files are intentionally absent from git.

### Final handoff: fallbacks and exceptions actually used

- Continued after the time-boxed repeatability investigation under Griffin's explicit instruction, retaining the unresolved limitation. No complete safe determinism fix was established, and no test, seed or budget was changed.
- Applied the authorized single-buffer sampling exceptions: exact64/64 whenever both sources exist, otherwise128 from the available source.
- No Goal7 single-change trick qualified for combination under the predeclared spread rule, so round2 had zero jobs. No required exploration family was cut for time; the mechanism fallback and the fallback forcing plain aim were not used.
- Used the written identical-zero-results exemption only for the hovering policy whose own floor and ceiling are both zero.
- For four movement results above the no-bias reference, ran the required500 tests at seed789 and reviewed physical trajectories instead of treating those scores as a stop. No extra score selected or replaced a result.
- Used the single available video example for the rare after failure type and no invented videos for zero-case types. Separate pose repeats are disclosed and never replace headline scores.
- Used the allowed separate episode-probability predictor for Goal8 because raw Q is not an episode-success probability; it learns only from counted practice. Necessary intermediate analysis traces use the explicit later-goal exception and remain local.
- Corrected one synthetic failure-rule fixture and one display-label assumption in an audit, preserving experimental thresholds and data. The camera chart legend was repaired before reporting. These were implementation/presentation repairs, not altered tests or results.

### Final handoff: open questions

- Why does learned aim correction often collapse early when a true-error oracle works well?
- Which remaining simulator operations prevent exact repetition during contact, including before learning?
- Does the movement gain hold with more independent seeds and on real hardware?
- Can a probability predictor trained on changing practice behavior provide useful early failure warnings?

These questions remain unanswered; the current work does not claim real-robot safety or performance. Every measured result and all chosen defaults are retained for review.


## 2026-11-15: Follow-up Phases A-G started

The finished Goals1–13 remain complete. FOLLOWUP_PLAN.md and the AGENTS.md report-path change were committed/pushed as c756efd before new work. Read the follow-up plan and standing rules; no planned check-in applies. New outputs/caches stay under results/takehome/followup. Initial interpretations and runtime estimates are recorded there and in the decision log. PhaseA reads saved data only; new episodes so far:0. Whole-project searched settings remain28.

## 2026-11-16: Follow-up Phase A complete

### 1. What I did (short list)
- Analysed all five saved aim confirmation final traces, their own 200-episode practice archives, and the saved original-policy repeat.
- Measured start, contact, first success and end errors separately; fitted fixed linear prediction probes from 1, 20 and 40 readings using practice only.
- Saved a preservation hash manifest, diagnostic data, tables, chart and new follow-up report.

### 2. What I found (the numbers, with how many test episodes and seeds)
Five saved 500-episode tests at seed456: contact error1.00–1.56mm, success error1.07–2.21mm, end error1.65–2.41mm, start0.953mm. Every seed's success error exceeded its paired starting error. End closer11.2–24.2%; contact step127–136. Longer linear history did not improve bias prediction. Old-policy saved repeat:86/99 successes below0.5mm bias,8/67 above1.5mm.

### 3. Checks: each check from the goal, marked pass or fail
PASS: all outside observations verified or corrected, including nonconstant directional dependence. PASS: first-success error paired to the same episodes. PASS: 1/20/40-reading prediction probes trained on saved practice only. PASS: one table and plain explanation plus chart. PASS: no new episodes; no original file edited; original report unchanged. PASS: all five seed records and30 probe records validated, chart visually reviewed after label fix.

### 4. Surprises or problems
The evaluator continues to step after first insertion, so final-step error exaggerates drift; success-time error still shows the problem. True bias remains imperfectly inferable from these fixed linear probes. The old-policy saved diagnostic repeat is303/500, not a replacement for the304/500 original floor. Exact physical repeatability remains limited as previously measured.

### 5. Decisions I logged
Eleven PLAN GAP FOLLOWUP entries fix preservation, exact report-prefix protection, approximate success labels, practice-only diagnostic fitting, contact and paired-success metrics, verdict priority, eligible selection, required integrated ideas versus optional stacks, setting caps, diagnostic seed789 and numerical defaults.

### 6. Practice episodes used, and settings tried so far
Follow-up0 practice,0 new full tests,0 new settings. Original totals remain29010 practice,409690 full tests,191 reset-only probes,28 searched settings. Saved diagnostic tests are reused, not counted again.

### 7. Plain-language explanation for Griffin (five sentences at most)
The old aim learner often inserts without correctly identifying the camera error. Its guess drifts away even before success. A longer history did not help the simple fixed prediction test. We will still test a learner with memory, because it can use the readings differently. Each change will get its own result, including failures.

### 8. Questions before the next phase
None. Continue to Phase B under the existing stopping rule.

## 2026-11-16: Follow-up Phase B checkpoint after first four settings

The first four settings completed all three development seeds, with200 practice and500 final test episodes per run. B_gate:297,309,294/500,60.00%±1.59points; B_pullback:307,301,282,59.33%±2.61; B_memory:364,313,308,65.67%±6.20; B_labels:266,204,236,47.07%±6.20. The first three verdicts are NO REAL DIFFERENCE against their matched old aim seeds; labels are HURT. No setting qualifies for extension yet. Labels improve paired success-time guess error by0.308mm, while memory worsens it by1.004mm; improved guesses on the successful subset alone did not improve overall insertion.

Verification: zero correction303/500 versus frozen307/500 floor passed the3-point tolerance. Two separately counted10-episode code trials passed (the first used the incremental-label implementation draft, the second the final direct-prediction implementation). Expanded37 synthetic control, provenance, verdict and reporting tests passed. A complete hash/mtime check passed for all9071 protected files, with no original-report append yet. First-wave12 runs used2400 practice episodes, plus20 in code trials; all four jobs finished normally in570–580seconds including three seeds and startup. Warm per-run times170–177seconds with four jobs together match the estimated aggregate throughput.

Resume from the running detached b_wave2 suite: B_fast, B_constant, B_within_grid and B_aim_fixed, seeds0–2. Fixed-camera references are measured before each fixed learner. The next wave should run B_heat_normal, B_aim_one_sided, B_heat_fixed and B_movement_fixed after this wave finishes, then B_heat_one_sided after its matched plain-aim base exists. Recollect summaries with python -B -m takehome.followup_collect after each wave; extend eligible settings and any newly required matched bases to0–4. Memory failed its isolated HELPED condition, so the optional memory+constant-fill Q-map stack cannot be launched. Keep all failures/skips in the scoreboard and finish/report/commit/push Phase B before launching C. C's predeclared20 applicable settings and five inapplicable rows are written in followup_c_design.md and followup_c_manifest.py but have not run.

## 2026-11-16: Follow-up Phase B interrupted by the mandatory ceiling tripwire

This report supersedes the historical resume instructions above. Phase B is not complete. No further experiment may start without Griffin's direction under the stopping rule.

### 1. What I did (short list)

- Tested isolated Phase B changes with four detached GPU jobs at a time and cheaper helpers handling routine runs, tables, charts and audits.
- Finished 37 experiments across 12 attempted settings; retained the partially completed fixed-camera aim run and all failures/skips.
- Stopped all experiment work when the aim-only ceiling tripwire fired. Saved the exact test, reference, traceback, consumed budget and halt state.
- Reviewed the 16 already-recorded physical verification tests, finished the partial report and scoreboard, and verified preservation before saving this interrupted-phase checkpoint to git.

### 2. What I found (the numbers, with how many test episodes and seeds)

Every completed run used 200 practice episodes and a final 500-episode test at seed 123. Three-seed normal-camera results were: gate 60.00% ± 1.59 points, pullback 59.33% ± 2.61, memory 65.67% ± 6.20, labels 47.07% ± 6.20, faster contact nudges 68.20% ± 2.42, constant-offset fill 52.53% ± 18.14, global heat map 58.87% ± 7.10, and within-episode grid search 65.53% ± 3.58. Labels HURT; constant fill was UNSTABLE; the other normal-camera changes were NO REAL DIFFERENCE. Label training improved paired success-time guess error by 0.308 mm despite reducing success. No normal-camera change qualified as HELPED or as a working aim corrector.

On three seeds, one-sided plain aim was PROMISING at 80.20% ± 5.72 against its 69.40% floor; fixed-camera movement was PROMISING at 90.13% ± 6.29 against its matched 62.80% floor. Fixed-camera heat map was NO REAL DIFFERENCE at 92.40% ± 4.13 against matched plain aim's 89.47% ± 4.43. These different camera conditions cannot replace the normal-camera headline. Fixed plain aim completed seeds 0 to 3 with 457, 422, 463 and 425 successes out of 500; its partial mean was 88.35% ± 4.25. Seed 4 then scored 256/256 at the 160-practice curve point against its own measured 470/500 ceiling, triggering the mandatory stop. Its final 500-episode test did not run.

### 3. Checks: each check from the goal, marked pass or fail

- PASS: completed experiments respected their 200-episode counters; the interrupted run consumed 160, and no test data was learned from.
- PASS: the zero-correction check was 303/500 versus the preserved 307/500 floor; both 10-episode code trials finished, with their different implementation versions disclosed.
- PASS: 37 synthetic tests and the recorded structural/data-origin checks passed; frozen base-policy audits passed.
- PASS: all 15 scoreboard rows are retained with allowed verdicts, every completed seed, aim-error metrics where applicable, and explicit failed/skipped reasons. The report embeds the full table and the reviewed chart marks the interrupted row partial.
- PASS: the 16 already-required movement verification tests at seed 789 had no wall-passing or invalid-number flags, and all were excluded from selection.
- PASS: after the halt, all 9,071 protected files passed hashes and timestamps; the original report remains byte-for-byte unchanged. Large trace/model/video artifacts are ignored.
- PASS: fresh seeds are unused; no Phase C-G experiment ran; all experiment workers stopped and new job launchers refuse the active halt state.
- FAIL: Phase B could not finish because the explicit aim-only ceiling tripwire fired. Required pending tests and five-seed extensions remain incomplete. The later phases and final full test suite are not claimed complete.

### 4. Surprises or problems

Fixed-camera plain aim at seed 4 exceeded the measured ceiling by six percentage points, beyond the fixed five-point threshold. The saved ceiling's first 256 episodes had 241 successes, so that prefix also differs from the triggering 256/256 test. The trigger recorded no wall-passing or invalid-number flags, but the cause remains unresolved. No retry, seed change, larger test, threshold change or silent reinterpretation was made. The budget records 160 practice episodes, while the last complete checkpoint is at 150; the already-collected batch through 160 is saved and must not be recollected if resumption is later authorized.

### 5. Decisions I logged

Logged the exact stopping rule and preserved the partial setting as DID NOT RUN with its four completed final results clearly separated from the triggering curve result. Retained all spent episodes and the original selection criteria. No new experimental choice was made after the tripwire. The existing PLAN GAP readings, fixed numerical defaults and prepared but unrun Phase C inventory remain documented.

### 6. Practice episodes used, and settings tried so far

Follow-up: 7,580 charged practice episodes, comprising 37 completed 200-episode runs, one partial 160-episode run and two 10-episode code trials; 110,584 full test-only episodes, including 8,000 required physical verification episodes; 12 attempted settings and 0 new reset-only probes. Whole project: 36,590 practice episodes, 520,274 full test-only episodes, 191 reset-only probes and 40 settings. The three unlaunched scoreboard rows consume no practice and do not count as attempted settings. The accounting audit has zero validation errors.

### 7. Plain-language explanation for Griffin (five sentences at most)

The old aim learner often succeeds without finding the camera error, and none of the completed normal-camera changes has fixed that convincingly. Labels improved the guess on successful episodes but made insertion worse. Two easier camera conditions look promising, but their confirmation is unfinished. One fixed-camera aim result was suspiciously high under the rule you set, so all experiments stopped. The original 80.2% result is unchanged, and there is no fresh-seed follow-up result yet.

### 8. Questions before the next phase

BLOCKING: Griffin's direction is required because the aim-only ceiling tripwire in CODEX_GOALS.md fired. Do not launch or retry experiments, clear the halt, use fresh seeds, or continue to Phase C automatically. This is the plan's explicit stopping case, not a planned check-in. Read TAKEHOME_FOLLOWUP_REPORT.md and results/takehome/followup/B/tripwire.json first.

## 2026-11-16: Follow-up Phase B complete after authorized rule change

### 1. What I did (short list)
Committed revised plans and reason before continuing. Ran the49-cell offset sweep first with four detached workers. Preserved the seed4 halt, restarted it from scratch, finished the one-sided heat map and all PROMISING extensions. Prepared but did not run mixed/C/D/E experiments.

### 2. What I found (the numbers, with how many test episodes and seeds)
Sweep242/256 tied at zero and(+0.3,0)mm. One-sided aim416,419,368,410,392/500,80.20%±4.24,HELPED versus69.40% floor; paired guess error worsened0.465mm. One-sided heat392,403,386/500,78.73%±1.72,NO REAL DIFFERENCE. Fixed aim457,422,463,425,492/500,90.36%±5.81; fixed movement433,432,487,495,499/500,93.84%±6.76; both NO REAL DIFFERENCE under the large matched-floor spread. No normal-condition aim fix worked.

### 3. Checks: each check from the goal, marked pass or fail
PASS: all49 sweep cells first,256 tests each,no learning. PASS: all16 B scoreboard rows retained,including halted/conditional skips. PASS: all PROMISING extensions completed. PASS: completed runs200practice/500test123,no test learning,frozenpolicies. PASS:41 resumption tests plus5 synthetic trick tests. PASS:36 extra physical checks have no wall/nan flags and do not enter selection. PASS:all41 old halted-seed files hash-identical, original-report/code/results unchanged, large artifacts ignored. PASS:fresh seeds unused,later experiments unrun, normal aim failure reported plainly.

### 4. Surprises or problems
The sweep did not show a constant offset better than zero; no higher ceiling is claimed. The restart reached98.4% and was checked under the revised rule. Large fixed-floor variation keeps the five-seed verdict conservative despite large mean gains. Better insertion on one-sided error still does not mean the aim guess found the true error.

### 5. Decisions I logged
Logged the authorized ceiling-rule change,mixed exception,exact splits,restart ownership,recovery definition,finite-grid caveat,D applicability and adapter contracts. Normal tripwire,seeds,success rule,test sizes and per-experiment budget remain fixed.

### 6. Practice episodes used, and settings tried so far
Follow-up9,180practice,153,512fulltests,0reset-onlyprobes,13settings. Wholeproject38,190practice,563,202fulltests,191reset-onlyprobes,41settings. Includes preserved160practice,two10-episode code trials,12,544sweep tests and18,000physical-check tests. PhaseA and synthetic checks spend no episodes.

### 7. Plain-language explanation for Griffin (five sentences at most)
No constant offset beat zero in the sweep. Learning aim helped insertion when error always pointed one way, but did not accurately find that error. The normal error case remains unresolved. The fixed-error runs are complete, including the authorized restart. Mixed tests will now measure calibration and per-attempt error together.

### 8. Questions before the next phase
None. Commit/push this phase and continue with mixed references and four methods per split, then C to G without a planned check-in.

## 2026-11-16: Mixed-condition checkpoint

B completion was pushed as b89fc5c. All9 mixed reference pairs finished with500floor+500zero-error tests, no practice, no invalid-number or wall flags. The detached mixed_plain_wave1 suite runs the six plain aim/movement settings first, four GPU processes maximum, then the six heat-map/joint settings must follow after their bases finish. Extend PROMISING settings and any required matched bases, premeasure missing references once per split before parallel jobs. Do not edit registry while its suite parent runs. C/D and E/G code preparation has not launched later experiments; fresh choice file remains absent.

## Mixed-condition progress checkpoint 2026-11-16T22:55:53-08:00

All five required plain-method extensions completed on choosing seeds0–4, test123,200 practice episodes each. Heat-map and joint comparisons remain in progress under mixed_more_wave2; the suite is the only registry/scoreboard writer.

- M_mostly_fixed_aim: 87.08% ± 4.73 points; HELPED; counts 426, 453, 436, 401, 461 out of500 per seed.
- M_even_aim: 79.88% ± 6.78 points; HELPED; counts 410, 346, 435, 416, 390 out of500 per seed.
- M_mostly_fixed_movement: 90.32% ± 5.04 points; HELPED; counts 443, 469, 486, 435, 425 out of500 per seed.
- M_even_movement: 87.64% ± 7.46 points; HELPED; counts 413, 464, 479, 447, 388 out of500 per seed.
- M_mostly_episode_movement: 88.16% ± 6.66 points; HELPED; counts 395, 479, 467, 435, 428 out of500 per seed.

Routine C/D inventories matched their declared original values. Root-reviewed combination and final-choice contracts passed on CPU (four stack guard checks, two choice checks), with no simulated episodes or fresh-seed use. C/D/E code and final reporting are prepared; touch methods remain unstarted.

## Follow-up mixed comparison complete, 2026-11-16T23:07:08-08:00

### 1. What I did (short list)

Completed all three prescribed calibration/per-attempt splits, each with plain aim, heat map, movement and joint learning. Measured all 15 matched floor/reference pairs. Extended all five PROMISING settings to five seeds. Generated recovery data, every-seed scoreboards and charts.

### 2. What I found (the numbers, with how many test episodes and seeds)

Five-seed plain movement: mostly fixed90.32%, even87.64%, mostly per-attempt88.16%, all HELPED. Plain aim:87.08% and79.88% HELPED in the first two splits,69.93% on three seeds in the third (NO REAL DIFFERENCE). No heat-map or joint addition helped; even and mostly-per-attempt heat maps and even joint were UNSTABLE. Every final result uses500 tests at123; the full report gives every seed and aim error.

### 3. Checks: each check from the goal, marked pass or fail

PASS all12 requested settings; PASS exact prescribed spreads and matched references; PASS all PROMISING rows extended; PASS200 practice per experiment and no test learning; PASS frozen policies and finite states; PASS required34 extra diagnostic checks and170 inspected examples; PASS complete28-row scoreboard, charts and recovery table; PASS fresh seeds remain locked.

### 4. Surprises or problems

Aim learning improved insertion in two mixed splits while its guess became farther from the truth at success. Heat-map even seed1 scored0/500, so success-time metrics are undefined rather than zero. Joint even seed1 fell below half its matched base. A chart CLI invocation lacked the package path; running the same reviewed module with python -m fixed it without any simulation or result change.

### 5. Decisions I logged

Previously locked mixed fractions, matched bases, shared initial-seed chart and unclipped recovery formula were retained. Report assembly now includes the complete generated table automatically. C/D inventory audits and combination/final-selection contracts are ready.

### 6. Practice episodes used, and settings tried so far

Mixed:9,200 practice episodes across46 experiments and12 settings. Follow-up totals:18,380 practice,302,720 full tests,25 settings. Whole project:47,390 practice,712,410 full tests,191 reset probes,53 settings.

### 7. Plain-language explanation for Griffin (five sentences at most)

A repeatable calibration error gives aim learning something reusable, but higher insertion success does not mean the guess found the true error. Movement learning helped in every mixed split. The heat map and joint additions did not improve their matched plain methods, and some seeds failed badly. Fixed and mixed tests are different models of calibration plus per-attempt error, not claims about which one represents every real setup.

### 8. Questions before the next phase

None. Commit/push this phase, then begin the20 applicable movement singles in Phase C, with five explained skips. No planned check-in.

## Phase C running checkpoint 2026-11-16T23:22:59-08:00

The five fill-up singles are complete and audited. The detached c_singles suite remains the single registry/scoreboard writer, with four GPU jobs in progress.

- C_fill_0: 373, 455, 349 /500; 78.47% ± 11.12 points; NO REAL DIFFERENCE.
- C_fill_10: 306, 295, 373 /500; 64.93% ± 8.44 points; HURT.
- C_fill_20: 378, 456, 375 /500; 80.60% ± 9.18 points; NO REAL DIFFERENCE.
- C_fill_80: 381, 435, 444 /500; 84.00% ± 6.81 points; NO REAL DIFFERENCE.
- C_fill_120: 383, 433, 439 /500; 83.67% ± 6.15 points; NO REAL DIFFERENCE.

The mixed phase is pushed as93cb72a. A separate cheaper-helper audit matched all mixed report numbers, all46 ledgers and all34 diagnostic checks. Reproduction instructions were reviewed; the report-assembler top-level output exception is now explicit. No fresh confirmation or touch runs have started.

## Follow-up Phase C complete, 2026-11-17 UTC

### 1. What I did (short list)

Ran all20 applicable movement singles and listed five inapplicable aim-dependent settings. Extended the only PROMISING screen, privileged critic, to five seeds. Preserved the screening snapshot, generated the full chart, and audited budgets, test provenance, frozen policies and success vectors.

### 2. What I found (the numbers, with how many test episodes and seeds)

No HELPED movement change. Privileged critic438,432,451,388,447/500:86.24%±5.05, NO REAL DIFFERENCE against matched five-seed movement83.36%±2.99. Three-seed screens compare the matched baseline84.20%±1.74. Smaller0.1mm cap298,318,355/500:64.73%, HURT. Every final test has500 episodes/test123.

### 3. Checks: each check from the goal, marked pass or fail

PASS all20 applicable values and five skips; PASS one change per base; PASS every seed shown; PASS only PROMISING extension finished; PASS all62 experiments200 practice, no test learning, frozen policies and finite final traces; PASS no above-reference diagnostics required; PASS complete scoreboard/chart/report; PASS fresh seeds and touch unstarted.

### 4. Surprises or problems

The initial hidden-information gain did not survive five seeds. One critic and0.4mm cap had collapsed seeds. The force penalty did not reduce mean peak force. No Phase C single qualifies for stacking.

### 5. Decisions I logged

Before E runs, audited that all three normal joint/order controls repeat original configurations: keep conservative attempt slots but deduct audited repeats from distinct setting totals. Require missing matched joint seeds before judging any promising order on five. Skip zero-fill/sibling-fill combinations because the latter has no episodes on which to act.

### 6. Practice episodes used, and settings tried so far

Phase C12,400 practice across62 experiments,155,672 full test episodes,20 settings. Follow-up30,780 practice,458,392 full tests,45 attempted distinct settings. Whole project59,790 practice,868,082 full tests,191 reset probes,73 distinct settings.

### 7. Plain-language explanation for Griffin (five sentences at most)

None of these movement changes reliably improved the plain method under the fixed rule. The apparent gain from extra simulator information faded when two more seeds were included. Several changes made results worse or caused one seed to collapse. The original result and all unsuccessful runs remain intact.

### 8. Questions before the next phase

None. Commit and push C, then run the18 applicable fixed-camera aim singles in Phase D with four detached GPU jobs and six explained skips.

## Phase D running checkpoint 2026-11-17T08:08:33.274903+00:00

C is pushed as41b7975. The detached d_singles suite is the only registry/scoreboard writer and runs up to four jobs. Completed fill-up screens:

- D_fill_0: 478, 498, 459/500; 95.67% ± 3.90 points; PROMISING.
- D_fill_10: 411, 29, 467/500; 60.47% ± 47.67 points; UNSTABLE.
- D_fill_20: 468, 436, 473/500; 91.80% ± 4.01 points; NO REAL DIFFERENCE.
- D_fill_80: 478, 487, 471/500; 95.73% ± 1.60 points; PROMISING.

Promising settings will be extended after singles finish. Final-confirmation safeguards passed10 synthetic tests after fixing a missing fake-process PID; no experiments were involved. Touch preparation passed5 synthetic tests, with no Phase F data inspection or simulation yet. Both touch entrypoints are gated on Phase E completion. Fresh choice remains absent.

## Resumption checkpoint 2026-11-17T08:22:30.542132+00:00

Active phase D; detached d_singles PID206395, suite parent is the sole registry/scoreboard writer. 10 of18 runnable settings finished; 59 follow-up setting slots attempted so far. Do not launch another suite or edit the registry until its parent exits. Then preserve the three-seed screening snapshot, extend every PROMISING row to0–4, close/report/commit/push D, and continue E.

Prepared while D runs: root reviewed final supervisor and reference CLI, cross-seed identical-outcome check, touch saved-pose diagnosis, blind-search controller/reference runner and hand-reference renderer. The10 final-safety contracts and5 touch contracts pass; these used no real episodes or fresh keys. F diagnosis and hand entrypoints refuse execution before E completion. No F data have been analyzed and no touch method has been tested. Torque/probe implementations and teacher training remain conditional work for F, not completed work. Source-backed contact notes are in takehome/followup_contact_model.md; F design is in takehome/followup_f_design.md.

Root inspected the first40 D above-reference checks and the saved trajectory plot:20,000 test-only episodes,200 bounded-control examples, no wall or NaN flags. More diagnostics are arriving and must be reviewed at D closeout. D/fill_failure_comparison.json describes the collapsed10-fill seed without claiming a causal training explanation. All original results and the80.2% headline remain unchanged; G/choice.json is absent.

## Phase D stalled-worker recovery, 2026-11-17 UTC

D_aim_cap_200um seed1 made no progress after08:20:19UTC at checkpoint90. Its ledger has100 reserved episodes and practice_0100.npz is complete. GPU is visible,10GiB RAM is available, otherjobs advance, and no invalid-number/tripwire message exists. Root preserved checkpoint90, ledger100, progress90 and SHA256 hashes of all10 saved practice batches under D/stalled_cap_recovery/, then sent SIGINT only to worker218939. Let d_singles finish its other work; it may report an ordinary failed worker. After it exits, resume this setting under a new suite name with the same directory/config/seeds, preserving all existing results and using saved batch100. Do not create a fresh seed or recollect any charged batch. The status is an operational recovery, not a scientific tripwire halt.
