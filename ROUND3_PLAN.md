# Round 3 plan: four ideas aimed at known weaknesses

Written 2026-11-17 for a coding agent (Claude Code) starting with no memory of earlier sessions. Everything needed is in this file and the files it names. Plain language on purpose. Nothing here has been run, so every expectation below is a guess to be tested.

## Background in one page

This project is a take-home for a robotics company. A simulated robot arm inserts a square peg into a square hole with 0.5 mm of clearance. A policy trained earlier (`results/policies/force_0.5mm_s2`, called "the old policy") succeeds about 61 percent of the time, because the reading of where the hole is has a hidden sideways error (the "camera error") of about 1 mm that is drawn fresh for every episode.

The assignment: improve that checkpoint with an off-policy, value-based reinforcement learning method, using at most 200 new practice episodes per experiment, compare with the original, and explain every choice. The reviewers care more about reasoning and honest analysis than the number.

What has been done and confirmed (do not redo it, do not change it):
- The method that works is the "plain movement corrector": the old policy stays frozen, and a small learned network adds a capped correction (0.2 mm per step) to the hand movement. It learns from a memory bank of past steps with four scorekeeper copies (a scorekeeper is a network that rates how good a move is, also called a Q-function or critic). Each experiment uses 40 fill-up episodes (the old policy acts with small random corrections to stock the memory bank) and then 160 learner episodes.
- Results on seeds never used for choosing: 80.2 percent (practice seeds 100 to 104, test seed 456) and 84.2 percent (practice seeds 200 to 204, test seed 1001), against a floor of about 61 percent and a no-camera-error reference of 93 to 95 percent.
- About 94 settings have been tried in total. None of 20 changes to the movement corrector helped on five seeds. An "aim corrector" (a learned shift of where the robot thinks the hole is) does not work when the camera error changes every episode, because the error cannot be worked out from the robot's sensor readings.

Four weaknesses this round targets:
1. Learning is unstable. Average success along the way dips (for example down to 61 percent around episode 120) before recovering, and runs end anywhere from 72 to 89 percent.
2. The success signal has no slope. Reward is 1 when the peg goes in and 0 otherwise, so the learner gets no "getting warmer" signal.
3. Sideways wiggling seems to help by accident. The failed aim corrector raised success in some conditions without finding the camera error. Saved data suggests its back-and-forth motion acts like a small search. This was never tested directly.
4. Nothing yet connects to what the company does, which is making learned models small and fast enough to run on robot hardware.

Where to look for detail, only if you need it: `TAKEHOME_REPORT.md` (the original study), `TAKEHOME_FOLLOWUP_REPORT.md` (the second round), `results/takehome/followup/scoreboard.md` (one row per setting), `takehome/` (all the code, including `learner.py`, `train.py`, `correction.py`, `episodes.py`, `budget.py`, and the `followup_*.py` runners).

## Rules for this round

1. Read `AGENTS.md` (the standing rules) and lines 1 to 60 of `CODEX_GOALS.md` ("Which file wins", "When to stop", "Definitions"). They apply in full, with the changes written in this section.
2. Work only on the git branch `takehome`. Never change existing code in `peg_insertion/`, `scripts/`, `tests/`, existing policies, anything outside `results/takehome/round3/` that already exists, `CLAUDE.md`, `PLAN.md`, or anything in `.claude/`. The 80.2 and 84.2 percent results and both existing reports stay as they are, except for the one clean-up in Task 0.
3. New code goes in `takehome/` with names starting `round3_`. New outputs go in `results/takehome/round3/`. The write-up is a new file at the top of the project, `TAKEHOME_ROUND3_REPORT.md`. Log every choice in `TAKEHOME_DECISIONS.md` and progress in `TAKEHOME_PROGRESS.md`.
4. Budget: 200 practice episodes per experiment, counted by the existing episode counter. Every episode that is learned from counts. Nothing is ever learned from a test episode.
5. Hidden simulator information (the true camera error, the true hole position) is never given to a learner or used in a reward. It may be used afterwards only to measure things.
6. Seeds, fixed now.
   - Choosing: test seed 123. First look on practice seeds 0 to 2. Anything that beats its base by more than one spread is rerun on practice seeds 0 to 4 before it is believed. Compare against the base on the same seeds.
   - Final check: practice seeds 300 to 304 and test seed 2002, which have never been used. Nothing runs on them until the final choice is written down.
   - Already used, never to be used for choosing or for the new final check: test seeds 456, 789 and 1001; practice seeds 100 to 104 and 200 to 204.
   - "Spread" means the standard deviation across practice seeds of the success rate at the end of the 200 episodes. Results are always taken at the end of the 200 episodes, or from the version chosen by a rule that uses practice episodes only (Task 1). Never pick a version by looking at test results.
7. Scoreboard. Keep `results/takehome/round3/scoreboard.md` and `scoreboard.csv`, one row per setting, added as soon as it finishes, including ones that do badly or fail. Columns: task, name, one plain sentence saying what it is, base and base result, each seed's result, average, spread, difference from base, verdict, average steps to insert, peak force, practice episodes used, and yes or no flags for "uses a hand-written routine" and "uses hidden information". Verdict words, in capitals: HELPED (beats the base by more than one spread, on five seeds), PROMISING (beats it on three seeds, not yet rerun), NO REAL DIFFERENCE, HURT (worse by more than one spread), UNSTABLE (at least one seed collapsed below half the base), DID NOT RUN (with the reason). Keep a short plain summary at the top, rewritten when each task finishes.
8. Limit: at most 30 new learning settings in this round. Keep the running count for the whole project (94 before this round) and state the total in the report.
9. Reuse the existing code for training, the episode counter, test runs, detached launching and the queue that runs four GPU jobs at once. Do not rewrite or refactor it, and do not write to the second round's registry or scoreboard in `results/takehome/followup/`. Make round 3 equivalents with the thinnest wrapper you can.

## How to work

Roles. The main session (Sonnet) is the workhorse: it runs everything, writes the code and keeps the records. Fable is the advisor, reached through the existing subagent `fable-debugger` (use `opus-debugger` if Fable cannot start, and note that once). Consult the advisor freely: whenever you fail at something, cannot explain an error after one honest attempt, are unsure how to proceed, or face a judgment call about what results mean. Also consult it at three fixed points: on the design of Tasks 2 and 3 before running them, on the design of Task 4 before running it, and on your written final choice and the report's conclusions before the fresh-seed runs. To consult: write a brief of at most 60 lines to `results/takehome/round3/advisor/brief_<n>.md` with the question, the exact evidence (numbers and file paths, not pasted logs), what you tried and what you need back. Tell the advisor to read that brief and only the files it lists, to ignore the instruction in its own definition to read `ESCALATION.md`, `PLAN.md` and `PROGRESS.md`, and to answer in at most 40 lines. Save the answer next to the brief and log one line per consultation in `TAKEHOME_DECISIONS.md`.

Keep going. A failure in one item must never stop the whole job. If something does not work after your attempt and one advisor consultation, park it: give it a scoreboard row marked DID NOT RUN with the reason, add it to a PARKED list in the state note, and move on to the next thing that does not depend on it. Revisit parked items once at the end. The only things that stop the job are the honesty tripwires in "When to stop" (episode budget passed, a test episode learned from, invalid numbers, original files changed) and a GPU that is not visible. Never work around those. A movement-type learner scoring above the no-camera-error reference is not a stop: test it once more on test seed 789, check for the peg passing through a wall, log it and continue.

Run in parallel. Keep four GPU jobs running whenever there is eligible work. Tasks 1, 2, 3 and 4 do not depend on each other, so queue them together. Only Task 5 (combining) and Task 6 (the final check) wait for the others. While the GPU is busy, do work that needs no GPU. Only one queue parent may write the round 3 scoreboard at a time, so put independent settings into one queue.

Save tokens. Experiments are plain Python and need no model while they run. Wait with a single command such as `sleep 540 && tail -n 3 <log>` and do not check more often than every 9 minutes. Never print whole logs, JSON files or source files; use `tail -n` with a small number, `grep`, or a short Python command that prints only what you need. Read each plan and rules file once. Keep a state note of at most 20 lines at the very top of `TAKEHOME_PROGRESS.md` under the heading "CURRENT STATE (ROUND 3)": what is running and its log name, what is queued, the PARKED list, open decisions. If your context is compacted or the session resumes, read only that note and the last log line before acting. Routine side tasks (a table, a chart, checking files exist) may go to a subagent on the cheapest available model with a narrow written task and a reply of at most 10 lines. Keep messages to Griffin short; put results in the scoreboard and report.

Running things. Every shell command starts with `cd /mnt/c/Users/Griff/Projects/peg-insertion && source scripts/env.sh && `. Confirm the GPU first with `python -c "import jax; print(jax.devices())"`, which must list a CudaDevice; if it does not, stop and say so. Start long runs detached. Keep trace, model and video files out of git. Commit and push to `origin takehome` whenever a task or a queue finishes.

## Task 0: clean up three stale sentences (no experiments)

`TAKEHOME_FOLLOWUP_REPORT.md` has leftover sentences from before the second round finished: the second paragraph says phases "D through G remain", and "Checks and limits" says the final test suite "has not run yet", the fresh seeds "remain unused" and the touch methods "remain unstarted". Check the saved records for what is true now and correct only those sentences. Change no numbers. Log the correction.

## Task 1: keep the best version, using practice data only

Problem it targets: weakness 1, the dips and the spread between runs.

Idea: during the 160 learner episodes, track how often the learner's own recent practice episodes succeeded. Use that, and nothing from any test, to decide which saved version of the learner to trust.

Try these as separate settings, each one change from the plain movement corrector:
- 1a. Pick at the end. Train exactly as now. At the end, use the saved version whose surrounding practice episodes had the highest success rate, instead of always using the last one. Costs no extra episodes and changes nothing during training.
- 1b. Roll back during training. After each batch of practice episodes, if the success rate over the most recent 20 learner episodes has fallen clearly below the best 20-episode stretch so far, restore the corrector network from that best point and continue. Choose the threshold before running and log it.
- 1c. Both together.

Things to get right: practice episodes include exploration noise, so their success rate is lower and noisier than a clean test. That is fine, since it is only used to rank versions. The window and threshold are fixed before seeing results.

Report: final success and spread against plain movement on the same seeds, and a stability measure: the lowest point of each run's learning curve after episode 80, and how far the final result is from that run's best curve point.

## Task 2: a reward the robot can feel

Problem it targets: weakness 2, no slope in the success signal.

Idea: add a small "getting warmer" reward built only from readings the robot has: the insertion depth reading and the wrist force and torque readings in the observation. No hidden simulator values.

Use the form that provably does not change which behaviour is best (called potential-based shaping): define a score for a situation from the readings, and add "discount times the score of the next situation, minus the score of this one" to the reward. The success test and the 1-or-0 success reward stay exactly as they are.

Try these as separate settings on the plain movement corrector:
- 2a. Score from depth only (deeper is better).
- 2b. Score from wrist torque only (less twisting is better).
- 2c. Both.
Then, because this idea came from trying to rescue the aim corrector:
- 2d. The best of 2a to 2c applied to the aim-only corrector in the normal camera condition (base: plain aim, about 64 percent on seeds 0 to 2).

Choose the sizes of the scores before running so the shaping reward over a whole episode stays smaller than the success reward, and log them. Known risk to check first from saved traces: the force readings are noisy and hit their maximum in about 41 percent of resting samples, so 2b may carry little signal. Report peak force as well as success, since rewarding low torque could also change how hard the robot pushes.

## Task 3: learn how to search, instead of where to move

Problem it targets: weakness 3, the accidental wiggle.

Step 3.0, settle the open question first (test only, no learning, no practice episodes). Take saved learners from the second round where an aim corrector scored well under a fixed camera error (for example `results/takehome/followup/B/B_aim_fixed/` and `D/D_aim_cap_200um/`) and replay 500 test episodes at test seed 123 four ways: (1) the full learner; (2) the guess held constant at its average value before contact; (3) the learner until first contact, then frozen; (4) the constant from 2 plus a small random back-and-forth of about 0.1 mm per step. If (2) and (3) lose the gain and (4) gets it back, the wiggle is what helps. Say plainly what was found.

Step 3.1, a hand-set wiggle as a control (test only, no learning, flagged as a hand-written routine): the old policy plus a fixed small sideways circular wiggle applied through the movement correction once the peg has made contact and depth has stalled, at three sizes (0.05, 0.1 and 0.2 mm). This shows what a constant wiggle does with no learning at all.

Step 3.2, learn the search. The learner outputs a few search settings instead of raw movement corrections, and a fixed routine in `takehome/` turns them into capped movement corrections: how wide to wiggle (0 to 0.3 mm), a steady sideways lean (two numbers, within the movement cap), and how much extra downward push (within the cap). The routine is zero before contact, so the system starts identical to the old policy. Same scorekeeper recipe, same 40 fill-up and 160 learner episodes, same counter.
- 3a. Search settings chosen every step.
- 3b. Search settings chosen once every 10 steps and held (fewer decisions to learn).
- 3c. The better of 3a and 3b, plus the plain movement correction added on top (more outputs).
Base for all three: plain movement corrector. Also compare with the best hand-set wiggle from 3.1, to show whether learning adds anything over a constant wiggle.

## Task 4: shrink the improved policy and measure what is lost

Problem it targets: weakness 4. This task is a measurement, not a new learning method, and it cannot be chosen as the final method.

Idea: the improved system is two networks (the frozen old policy plus the movement corrector). Compress it into one small network, lower its number precision, and report success against size and speed.

Steps:
1. Teacher: the saved final plain movement learners from the second round's fresh-seed runs, `results/takehome/followup/G/plain_movement/seed_200` to `seed_204` (file `model_0200.pkl` in each, with the old policy). Confirm each teacher reproduces its recorded result on test seed 1001 before using it; say so if it does not, given the simulator's known lack of exact repeatability.
2. Training data: only that run's own recorded practice steps (`practice_*.npz` in the same folder). For each recorded reading, compute what the teacher would do with no exploration noise, and train the student to copy that action. This uses no new episodes.
3. Students, one network taking the normal 16 readings and giving the 3 movement numbers: the same size as the old policy, about half the size, and about a quarter.
4. Precision: full (32-bit), half (16-bit), and 8-bit. Simulating 8-bit by rounding the weights is acceptable; say clearly that it is simulated and not a real 8-bit runtime.
5. Measure for each student: success on 500 test episodes (test only), number of weights, size in bytes, and time per decision on the laptop processor with a batch of one, labelled as indicative only.
6. Known risk: a student trained only on recorded steps can drift into situations the recordings never covered and do worse there. If the best student is more than 5 points below its teacher, run one extra variant, clearly labelled as using extra practice: up to 200 more counted episodes in which the student acts and the teacher's action is recorded as the label, then retrain. Report both.

Report: one table and one chart of success against size, with the teacher and the original policy as reference lines, for each precision.

## Task 5: combine what helped

Only settings from Tasks 1 to 3 that earned HELPED on five seeds may be combined, at most three combinations, skipping any that cannot work together. If nothing earned HELPED, say so and skip this task.

## Task 6: final check and write-up

1. Choose the final learning method by the fixed rule: highest average on test seed 123 with practice seeds 0 to 4; among those within one spread of the top, the one with the fewest changes from plain movement. Plain movement is a candidate. A method that uses a hand-written search routine with no learned part cannot be chosen. Task 3.2 methods can be chosen, and the report must say they contain a designed search routine. Write the choice and reason down, then consult the advisor, then continue.
2. Run the chosen method and plain movement on practice seeds 300 to 304 with test seed 2002, and measure the floor and the no-camera-error reference on test seed 2002. If the chosen method is plain movement, run it once and say so.
3. Apply Task 4's shrinking to the chosen method's fresh-seed learners if it differs from plain movement and time allows; otherwise say it was done on plain movement only.
4. Write `TAKEHOME_ROUND3_REPORT.md` in plain language with no em dashes: the four weaknesses, what each task tried and found, the full scoreboard, the Task 3.0 answer, the Task 4 table and chart, the final check, the total number of settings tried in the whole project, and the limits. Say plainly whether anything beat plain movement or not, and that later rounds had far more tuning than the first.
5. Run the full test suite, confirm the original code, policies and results are unchanged, commit and push.
6. Give Griffin a short final summary: the chosen method and its fresh-seed result, how it compares with plain movement on the same seeds, which ideas HELPED, the size and speed table's headline, what was parked and why, how many advisor consultations were used, and what to read first.
