# Take-home progress

## Current status: Goals 1-9 complete; Goal 10 next (2026-11-15)

Final movement method and80.20% confirmation headline remain unchanged. Failure categories cover all495 final-method failures and197 before-repeat failures, with every seed shown. All13 illustrative videos passed audits and visual review. Next measure the three other policies' own references, then train the unchanged recipe on seeds0-2. Total28 settings,22,210 practice episodes and297,970 full test-only episodes. Continue through Goal13 with no cuts; binaries remain local and repeatability remains a documented limitation.

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
