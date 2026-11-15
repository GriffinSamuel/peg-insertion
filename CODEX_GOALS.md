# Goals for the coding agent, one per phase

The agent works through these goals in order. Each block says what to achieve, what to hand back, and how to know it is done. The background and reasons are in `TAKEHOME_PLAN.md`. The standing rules are in `AGENTS.md`.

## Autopilot rules

There is one planned check-in with Griffin: after Goal 5. Goals 1 to 5 are cheap and show whether the setup works. Everything after Goal 5 is long compute and runs without waiting.

Self-checks replace the other stops:
- After each goal, test your own work against that goal's "Done when" list and write the report (headings below) into `TAKEHOME_PROGRESS.md`. If every check passes, go straight on.
- If a check fails, try to fix the cause once. If it fails again, apply the fallback written in that goal. If there is no fallback, halt.

Tripwires. Halt and wait for Griffin only if one of these happens:
- the GPU is not visible, or runs produce invalid numbers
- the zero test in Goal 3 fails (the wrapper changed the old policy's behaviour)
- the episode counter was passed, or a test was learned from
- a result is better than the no-camera-error ceiling by more than 5 points, or all seeds give identical results (both suggest a bug)
- any change would be needed outside `takehome/` and `results/takehome/`

Choosing the best setting without a human, fixed in advance:
- pick the setting with the highest average success on the development test
- if two settings are within the seed-to-seed spread of each other, pick the one with fewer tricks switched on
- write down the choice and the reason before running the confirmation

Long runs: start them detached so they survive a closed terminal, the way `scripts/train.sh` does. Skip any run whose results file already exists, so the work can be resumed after an interruption.

## How to report after every goal

Use exactly these headings:

1. What I did (short list)
2. What I found (the numbers, with how many test episodes and seeds)
3. Checks: each check from the goal, marked pass or fail
4. Surprises or problems
5. Decisions I logged
6. Practice episodes used, and settings tried so far
7. Plain-language explanation for Griffin (five sentences at most)
8. Questions before the next phase

## Goal 1: measure the starting line

Achieve: know how good the starting policy `force_0.5mm_s2` is, with and without the camera error.
Hand back: `results/takehome/baseline.json` with both success rates from 500 test episodes each, test seed 123.
Done when:
- the normal result is within 5 points of the recorded 62.8 percent
- the no-camera-error result is within 5 points of the recorded 94.0 percent
- the first command run confirms the GPU is visible to JAX

## Goal 2: build the aim-corrected environment

Achieve: a wrapper where a learner nudges a running guess of the camera error (two numbers), the guess is subtracted from the hole reading, and the frozen old policy acts on the corrected reading.
Hand back: code in `takehome/`, with tests.
Done when:
- the guess starts at zero each episode and cannot leave its cap
- reward is 1 at first insertion and 0 otherwise, the episode ends for the learner at success, and a time limit is marked differently from a failure
- the old policy's network weights are provably unchanged after a run

## Goal 3: prove the mechanism before any learning

Achieve: evidence that correcting the aim is enough to help, before spending effort on learning.
Hand back: two numbers in `results/takehome/mechanism_check.json`.
Done when:
- zero test: with the nudge always zero, success is within 3 points of the Goal 1 normal result on the same test seed
- cheat test: with the true camera error fed in as the guess, success is within 5 points of the Goal 1 no-error result
Fallback if the cheat test fails twice: do not try to rescue it. Record the failure, make the movement corrector the main method for Goals 4 and 5, keep the aim corrector as a reported negative result, and continue. If the zero test fails, that is a tripwire.

## Goal 4: build the learner

Achieve: a memory bank, a scorekeeper (several copies, layer normalisation, predicts chance of success), an aim corrector network, and a learning loop that studies the memory bank many times per new step.
Hand back: code and tests in `takehome/`.
Done when:
- a 10-episode trial run finishes with no errors and no invalid numbers
- the episode counter refuses to collect episode 201
- one command starts a full 200-episode experiment from a named policy and seed
- each study batch is half fill-up data and half new data, the correction numbers are rescaled before the scorekeeper sees them, and the old policy's input scaling is reused
- every trick listed in Goal 7 can be switched on or off with a setting, without editing code

## Goal 5: run the main experiment (the one check-in with Griffin is after this)

Achieve: an honest measurement of what the aim corrector does in 200 episodes.
Hand back: learning curves and a results file for 5 seeds, with about 40 fill-up episodes and 160 practice episodes each, tested every 20 practice episodes on separate test episodes.
Done when:
- all 5 seeds finished and every seed is shown, not only the average
- the final result is compared with the floor and ceiling from Goal 1
- the report says plainly whether it improved, stayed the same, or got worse
There is no target number. A flat or negative result reported honestly counts as done.
After this goal, stop once and show Griffin: the Goal 1 floor and ceiling, the Goal 3 checks, the learning curves, how long one run took, and the estimated time for everything that remains. Continue when he says go.

## Goal 6: comparisons at the same budget

Achieve: know how the aim corrector compares with the usual approach.
Hand back: one table and one chart with five lines: original policy, aim corrector only, movement corrector only (small capped corrections to hand movements, same scorekeeper recipe, same 200 episodes), both together (one learner outputs the aim nudge and the movement correction, one scorekeeper, same 200 episodes), and a clearly labelled outside-the-budget reference trained with true labels.
Done when: every method used the same test seed and the same number of seeds, the combined learner starts from zero correction so it begins identical to the original policy, the outside-the-budget one is labelled as such everywhere it appears, and the report says which of the three learners did best and what each part added.

## Goal 7: exploration round, then one confirmation run

Achieve: find the best settings through wide experimentation, and report a number that was not picked by peeking.
Explore, one change at a time from the main experiment:
- fill-up share: 0, 10, 20, 40, 80, 120 of the 200 episodes
- aim and movement together: joint (one learner, one scorekeeper), side by side (two learners, two scorekeepers, same time), one after the other (aim then movement, and movement then aim)
- scorekeeper briefed first or not
- scorekeeper sees true camera error in training or not
- several scorekeepers or one
- how many times the memory bank is studied per new step
- size caps on the aim nudge and the movement correction
- a final decider: each step, rate several options with the scorekeeper (no correction, aim only, movement only, both, a few small variations) and act on the best, versus acting on the learner output directly; use only a scorekeeper that does not need the true camera error
- hand over near the hole (old policy alone on the approach, learner active only near the hole)
- looser cap in contact, tighter in free space
- movement cap sized by the old policy's own uncertainty
- scorekeeper trained as a yes or no predictor of success
- looking several steps ahead when crediting moves
- a floor on the scorekeeper's ratings from real outcomes
- a penalty for pushing too hard, with peak force reported
- fill-up episodes from sibling policies
Search in three rounds, not every combination: (1) screening, each trick on by itself; (2) combining, only the few tricks that clearly beat the seed-to-seed spread; (3) confirmation, the single best setting on new seeds.
Hand back: one exploration table with every setting tried, all seeds shown, and the seed-to-seed spread next to each result.
Done when:
- all choosing used one fixed set of practice seeds and one development test seed
- each experiment used its own 200 practice episodes
- the total number of settings tried is stated
- differences smaller than the seed-to-seed spread are marked as not a real difference
- the best setting is chosen by the fixed rule in the autopilot rules, with the choice and reason written down first
- the chosen setting is then run once on new practice seeds and a new test seed never used for choosing, and that confirmation result is reported as the headline even if it is lower than the exploration result

## Goal 8: check the scorekeeper's honesty

Achieve: know whether the scorekeeper's confidence can be trusted.
Hand back: a chart of predicted chance of success against actual success, a list of confident-but-failed moments, and the step where confidence drops in failed episodes.
Done when: the chart is built only from test episodes, and the report says in one sentence whether the scorekeeper is overconfident, underconfident, or about right.

## Goal 9: failure cases

Achieve: a clear picture of how it still fails.
Hand back: counts of each failure type before and after (never touched the hole, stuck on the rim, ran out of time, corrected the wrong way), and a few videos per type showing the running guess next to the true error.
Done when: every failed test episode is assigned to exactly one type, and the counts add up.

## Goal 10: check it is not a one-off

Achieve: know whether the result holds on other policies.
Hand back: the main experiment repeated on `force_0.5mm_s4`, plus one position-only policy and one policy that hovers and never inserts.
Done when: all three are reported, including the ones expected to do poorly, with a sentence on why each behaved as it did.

## Goal 11: different camera problems

Achieve: know which kinds of camera error the corrector can and cannot fix.
Hand back: one table and one chart covering the seven problems in the plan, each with floor, test-only, retrain, and ceiling.
Done when: the new errors live only in `takehome/` code, each retrain used its own 200 episodes, and the height error is included even though it is expected to fail.

## Goal 12: write-up

Achieve: a report Griffin can send and defend.
Hand back: `TAKEHOME_REPORT.md` with the question, the idea, the method, all results with every seed, the charts, failure cases, every algorithm choice and its reason, the number of settings tried, the limits, the budget assumption, how AI assistance was used, and one command to rebuild each figure.
Done when: every number in the report can be traced to a file in `results/takehome/`, and nothing is claimed that the results do not show.

## Goal 13: final checks

Achieve: confidence that the work is clean and repeatable.
Done when:
- all old and new tests pass
- `git diff main -- peg_insertion scripts tests results` shows no changes outside `results/takehome/`
- one fresh rerun of the main experiment matches the reported result within normal seed-to-seed spread
- a final summary is written for Griffin listing what to read first, every fallback that was used, and every question left open
