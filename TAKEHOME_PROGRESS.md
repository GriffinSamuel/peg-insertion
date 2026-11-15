# Take-home progress

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
