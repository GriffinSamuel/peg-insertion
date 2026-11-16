# Take-home progress

## Current status: Goal 5 complete, planned check-in (2026-11-15)

The revised instructions are active and the old sampling pause is resolved. Goals 1–5 pass. The five designated runs average 65.16% success with 7.74 percentage points of sample standard deviation, compared with the 61.4% floor and 94.4% ceiling. The small average gain does not establish reliable improvement. All nine benchmark runs finished and are retained. Practice used: 1810 episodes across separate bounded experiments, including the ten-episode code trial; learning settings tried: 1. Stop here under CODEX_GOALS.md's planned Goal 5 check-in. No Goal 6 work or confirmation run has started. Read TAKEHOME_REPORT.md for the complete check-in, including timing estimates and proposed cuts.

## Current status after revised instructions, 2026-11-15

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
