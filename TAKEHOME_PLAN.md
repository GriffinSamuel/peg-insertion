# Take-home plan: improve a peg insertion checkpoint in 200 episodes

Written 2026-11-15 for the General Instinct take-home. This file is the plan for a coding agent (Codex) to follow, one phase at a time. It is written in plain language on purpose. It explains what and why. The exact numbers, seeds, definitions and the stopping rule are in `CODEX_GOALS.md`, and that file wins wherever the two differ.

## The assignment in one paragraph

Take one of the trained policies in this project and make it succeed more often, using an off-policy, value-based reinforcement learning method, with at most 200 new practice episodes. Compare against the original policy, show learning curves and failure cases, and explain every algorithm choice. The reviewers care more about the reasoning than the final number.

## The idea in one paragraph

The robot sees the hole through a camera reading that is slightly wrong. An earlier test in this project showed that when that error is removed, the half-working policies succeed 94 to 100 percent of the time. So the old policy already knows how to insert the peg and is mostly aiming at the wrong spot. We keep the old policy frozen and train a small new network, the "aim corrector", that keeps a running guess of the camera error (two numbers) and fixes the reading before the old policy sees it. A second new network, the "scorekeeper" (a Q-function), rates the corrector's choices and learns from a memory bank of past steps (a replay buffer). We also check whether the scorekeeper's confidence is honest and use it to explain failures.

## Ground rules for the agent

1. Work on a new git branch called `takehome`. Do not change the existing environment code, trained policies, or anything already in `results/`.
2. Put all new code in a new folder `takehome/` and all new outputs in `results/takehome/`. The three log files (`TAKEHOME_DECISIONS.md`, `TAKEHOME_PROGRESS.md`, `TAKEHOME_REPORT.md`) sit at the top of the project. Commit and push to the `takehome` branch after every goal.
3. Every choice that is not written here goes into a new file `TAKEHOME_DECISIONS.md` as one line with the reason, the same way `DECISIONS.md` works.
4. Keep a dated log of what was done in `TAKEHOME_PROGRESS.md`.
5. Count every practice episode. The total used for learning, including episodes where the old policy runs just to fill the memory bank, must never pass 200 per experiment. Test episodes that are not learned from do not count, and the write-up must say so.
6. There is one planned check-in with Griffin, after Phase 5. Everywhere else, check your own work and keep going. The single stopping rule is at the top of `CODEX_GOALS.md`.
7. Use the existing setup: run inside WSL, start with `source scripts/env.sh`, and reuse `peg_insertion/rollout.py` for loading policies and running test episodes.

## Phase 0: before any code (Griffin does this)

1. Decided on 2026-11-15: fill-up episodes count toward the 200. Every episode that is learned from counts, including ones where the old policy runs just to fill the memory bank. State this assumption in the write-up. Emailing Guanming to confirm is optional.
2. Pick the starting policy. First choice is `force_0.5mm_s2` (63 percent now, 94 percent with the camera error removed). Second is `force_0.5mm_s4` (85 and 96).

## Phase 1: measure the starting line

1. Test the original policy on 500 fresh episodes with a fixed test seed. This is the floor.
2. Test it again with the camera error switched off. This is the ceiling.
3. Save both numbers. Every later aim-only result should land between them. Learners that also correct the movement can honestly go above the ceiling.

## Phase 2: build the aim-corrected environment

1. Write a wrapper around the existing environment and the frozen old policy.
2. Each step, the learner outputs a small nudge to a running guess of the camera error. The guess starts at zero every episode and is capped at a few millimetres.
3. The wrapper subtracts the guess from the hole reading, gives the corrected reading to the old policy, and runs the old policy's action in the simulator.
4. The learner sees the normal observation plus its own running guess.
5. Reward is 1 the first time the peg is inserted and 0 otherwise. The episode ends for the learner at success. Running out of time is marked as a time limit, not a failure.

## Phase 3: prove the mechanism before any learning

1. Zero test: with the nudge always zero, the wrapper must give the same success rate as the original policy, within 3 points.
2. Cheat test: feed the true camera error as the guess. Success should rise to about the ceiling.
3. If the cheat test does not reach near the ceiling, the idea does not work as designed. Record it, switch the main method to the movement corrector, and continue. If the zero test fails, halt.

## Phase 4: build the learner

1. A memory bank that stores every step: situation, nudge, reward, next situation, and whether it ended.
2. The scorekeeper: several small copies with layer normalisation, trained to predict the chance of success.
3. The aim corrector: a small network trained to pick nudges the scorekeeper rates highly.
4. A learning loop that studies the memory bank many times for each new step collected.
5. Built in from the start: whenever both fill-up data and new data exist, each study batch is half of each (when only one kind exists, the batch comes entirely from that kind); the correction numbers are rescaled before the scorekeeper sees them so small corrections are not drowned out; the old policy's own input scaling is reused.
6. A short test that the whole loop runs end to end on a tiny budget.

## Phase 5: run the main experiment

1. Split the 200 episodes: about 40 where the old policy runs with small random nudges to fill the memory bank, then about 160 where the learner practises. Collect in small batches so the learner can improve between batches.
2. Every 20 practice episodes, pause and test the current learner on separate episodes that are not learned from. These points make the learning curve.
3. Repeat the whole 200-episode experiment with at least 5 different random seeds.
4. The one check-in: show Griffin the floor and ceiling, the mechanism checks, the first learning curves, how long one run took, and the time estimate for the rest. Continue when he says go.

## Phase 6: comparisons at the same 200-episode budget

1. The original policy (floor) and the no-error test (ceiling).
2. The standard approach: the same scorekeeper recipe, but the learner corrects the robot's hand movements directly. This is the main comparison.
3. Both together: one learner that outputs the aim nudge (two numbers) and a small capped correction to the hand movement (three numbers), rated by one scorekeeper, with the same 200 episodes. Comparing aim only, movement only, and both shows what each part adds.
4. A reference that cheats: a network trained to predict the camera error directly from true labels. It shows what is possible with information a real robot would not have.

Which of the three learners becomes the final product is decided by these results, not in advance.

## Phase 7: exploration round, then one confirmation run

The aim is to try many settings before deciding what is best, without fooling ourselves.

The honesty rule for this phase: trying many settings and reporting the best one is like sitting an exam many times and reporting the top score. So choosing and reporting are kept apart.
1. Choosing: every setting is compared using one fixed set of practice seeds and one "development" test seed.
2. Reporting: once the best setting is chosen, run it one more time on new practice seeds and a new test seed that were never used for choosing. That confirmation run is the headline number.
3. Every experiment still gets its own 200 practice episodes. Log every setting tried and report the total count.
4. A difference smaller than the spread between seeds is not treated as real.

Things to explore, changing one at a time from the main experiment:
1. Fill-up share: 0, 10, 20, 40, 80 and 120 of the 200 episodes spent filling the memory bank.
2. How the aim part and the movement part work together:
   - joint: one learner outputs both, one scorekeeper rates both
   - side by side: two learners, each with its own scorekeeper, learning at the same time
   - one after the other: aim first for part of the budget and then movement, and the reverse order
3. Scorekeeper briefed on the old policy's outcomes first, versus not.
4. Scorekeeper allowed to see the true camera error during training, versus not.
5. Several scorekeeper copies versus one.
6. How many times the memory bank is studied per new step.
7. The size caps on the aim nudge and on the movement correction.
8. A final decider: at each step, line up several options (no correction, aim only, movement only, both, and a few small variations), let the scorekeeper rate each one, and act on the best rated. Compare with acting on the learner's output directly. The decider only works with a scorekeeper that does not need the true camera error, because that is not available when the robot is really running.

More tricks borrowed from the papers, each switched on by itself first:
9. Hand over near the hole: the old policy does the approach alone and the learner only switches on close to the hole.
10. Looser cap in contact, tighter cap in free space.
11. Movement cap sized by the old policy's own uncertainty.
12. Scorekeeper trained as a yes or no predictor of success.
13. Looking several steps ahead when crediting moves.
14. A floor on the scorekeeper's ratings taken from real outcomes.
15. A penalty for pushing too hard, with peak force reported.
16. Fill-up episodes from sibling policies instead of one.

How to search without trying every combination (there are far too many):
- Round 1, screening: switch each trick on by itself and compare with the main experiment.
- Round 2, combining: take only the few tricks that clearly helped (beyond the seed-to-seed spread) and try their combinations.
- Round 3, confirmation: the single best setting, on new seeds and a new test seed.

After Round 2, choose the best setting by the fixed rule (highest average on the development test, and the simpler setting when two are within the seed-to-seed spread), write the choice down, then do the confirmation run.

## Phase 8: check the scorekeeper's honesty

1. Collect its predictions on test episodes and compare each confidence level with how often those episodes really succeeded. Plot predicted against actual.
2. List the moments where it was confident and the robot still failed.
3. In failed episodes, find the step where its confidence dropped sharply.
4. Track how honest it is over the course of training.

## Phase 9: failure cases

1. Sort failures into types: never touched the hole, stuck on the rim, ran out of time, corrected the wrong way.
2. Count each type before and after.
3. Render a few videos of each type with a new script in `takehome/` that reuses the existing video code, showing the running guess next to the true error.

## Phase 10: check it is not a one-off

1. Repeat the main experiment on the second policy.
2. Try one position-only policy and one policy that hovers and never inserts. These are expected to do poorly. Report them as limits of the method.

## Phase 11: different camera problems

The goal is to show the aim corrector fixes more than the one kind of camera error it was built around. Add each new error inside the `takehome/` wrapper, on top of the reading, so the original environment code stays untouched.

Camera problems to try:
1. Bigger constant offset: the same kind of error but two and four times larger.
2. One-sided offset: the camera is always off in the same direction, like a bad calibration.
3. Drifting offset: the error changes slowly during an episode.
4. Shakier reading: more random jitter from step to step.
5. Stretched or rotated reading: the error depends on where the peg is.
6. Late reading: the robot sees where things were a few steps ago.
7. Height error: the reading is wrong in the up and down direction. The corrector only fixes sideways error, so this one is expected to fail and shows a limit.

For each problem, run two tests:
1. Test only: take the corrector trained on the normal error and test it on the new problem with no extra practice. This shows how far it stretches.
2. Retrain: give it a fresh 200-episode experiment on the new problem. This shows whether the method itself can handle it.

For each problem also measure the floor (old policy alone under that problem) and the ceiling (no camera error), so every result is read against both. Make sure the cap on the running guess is large enough for the bigger offsets. Present one table and one chart: camera problem on one side, floor, test-only, retrain and ceiling on the other. Say plainly which problems it fixed, which it partly fixed, and which it could not.

## Phase 12: write-up

1. A report with the question, the idea, the method table, results with all seeds shown, learning curves, the honesty plot, failure cases, and the camera problems table.
2. A section on every algorithm choice and why, taken from `TAKEHOME_DECISIONS.md`.
3. How many settings were tried in total, stated honestly.
4. Limits: simulation only, which policies it cannot fix, what the budget assumption was.
5. How to reproduce every figure with one command each.
6. A statement of how AI assistance was used.

## Phase 13: final checks

1. All old and new tests pass.
2. The original results and policies are unchanged.
3. One clean rerun of the main experiment from scratch gives the same result.
4. Write a final summary for Griffin: what to read first, every fallback used, and every open question. Griffin reads the whole report and can explain every line before sending.

## How to run this with Codex

Open Codex inside WSL in this project folder. Give it this first message:

    Read TAKEHOME_PLAN.md, README.md, DECISIONS.md and the code in peg_insertion/. Work through the goals in CODEX_GOALS.md in order, following its autopilot rules. Check in with me once, after Goal 5, then continue to the end. Log every decision and write a report after every goal.

If Codex is interrupted, start it again and say: read TAKEHOME_PROGRESS.md and continue from the first goal that is not finished. At the end, read the final summary and the report, and ask it to explain anything unclear in plain language.
