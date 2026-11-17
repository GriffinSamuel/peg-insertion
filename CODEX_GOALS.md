# Goals for the coding agent, one per phase

The agent works through these goals in order. Each block says what to achieve, what to hand back, and how to know it is done. The background and reasons are in `TAKEHOME_PLAN.md`. The standing rules are in `AGENTS.md`.

## Which file wins

If two files disagree, `AGENTS.md` wins over this file, and this file wins over `TAKEHOME_PLAN.md`. The plan explains the reasons; this file sets the exact numbers.

## When to stop (the only stopping rule)

Stop and wait for Griffin in exactly four cases:
1. The planned check-in after Goal 5.
2. A tripwire fires (list below).
3. A "Done when" check still fails after one honest attempt to fix the cause, and that goal has no written fallback.
4. A problem in the plan that cannot be settled without changing the budget, the test, a test seed, or the success rule, or without writing outside the allowed places.

Everything else is not a stop. If an instruction is unclear, missing, or two parts disagree: take the most cautious reading, log it in `TAKEHOME_DECISIONS.md` starting with PLAN GAP, list it in the next report, and continue. Wherever the plan names a setting without a number (a distance, a split, a count), choose one sensible value, log it, and continue. Report heading 8 is information only; it does not pause the work unless the line starts with BLOCKING, and BLOCKING is allowed only for the four cases above.

After each goal, test your own work against that goal's "Done when" list and write the report (headings below) into `TAKEHOME_PROGRESS.md`. If every check passes, commit, push, and go straight on.

Tripwires:
- the GPU is not visible, or runs produce invalid numbers
- the zero test in Goal 3 fails (the wrapper changed the old policy's behaviour)
- the episode counter was passed, or a test episode was learned from
- an aim-only learner scores more than 5 points above the measured ceiling for that same policy and camera problem, on a policy whose ceiling is above 50 percent (the aim corrector cannot honestly beat a perfect aim)
- every seed of a learning run gives exactly the same outcome on every single test episode (this check does not apply to runs with no learning, such as the floor, the ceiling, the zero test and the cheat test, or to policies that score 0 percent both with and without the camera error)
- any change would be needed outside the allowed places in `AGENTS.md` rule 2

Also not a tripwire (added 2026-11-16 after the follow-up halt): an aim-only learner scoring above the ceiling in a camera condition where the error is consistent from episode to episode ("fixed for the whole run" or one-sided). The ceiling is the old policy with zero camera error, and saved data shows the old policy has its own sideways lean (it succeeds more often when the camera error points one way than the other), so a learned constant offset can honestly beat it there. Treat it like the movement case below: check it, log it, and continue. In the normal condition, where the error is redrawn every episode, the aim-only ceiling tripwire still applies.

Not a tripwire: a movement, combined, or final-decider learner scoring above the ceiling. Those change the robot's own movements, so they can honestly beat it. When it happens, check it instead of stopping: test again on one extra test seed, look at a few episodes for anything unphysical (the peg passing through a wall), log what you found, and continue.

## Definitions used in every goal

Words:
- Practice episodes: all episodes that are learned from. 200 per experiment, enforced by the counter. They come in two kinds: fill-up episodes (the old policy acts, with small random values on whichever outputs the learner controls, to stock the memory bank) and learner episodes (the learner acts).
- Main experiment: the Goal 5 setup. Aim corrector only, exactly 40 fill-up episodes then 160 learner episodes.
- Final method: the setting that wins Goal 7 and is confirmed there. Goals 8, 9, 10, 11 and 13 all use the final method, unchanged and without further tuning. "Before" means the original policy and "after" means the final method. Charts pool all seeds and also show each seed.
- Setting: one distinct configuration of the learner. The running count of settings tried covers Goals 5, 6 and 7. Goals 10 and 11 reuse the final method on new conditions and are listed separately.
- Spread: the standard deviation, across seeds, of the success rate at the end of the 200 episodes.

Seeds and test sizes, fixed in advance:
- Development test: test seed 123, 500 test episodes. Used for every result in Goals 5, 6 and 7 rounds 1 and 2, and for all choosing.
- Learning curves: the bottom axis is total practice episodes used, from 0 to 200, fill-up included. Point 0 is the original policy. After fill-up ends, test every 20 practice episodes with 256 test episodes (seed 123), and test the final point at 200 with the full 500.
- Practice seeds: 0 to 4 for anything reported. Goal 7 round 1 (screening) uses seeds 0 to 2 to save time; round 2 uses 0 to 4. Goals 10 and 11 use seeds 0 to 2.
- Confirmation: practice seeds 100 to 104 and test seed 456 with 500 test episodes. Nothing may be run on these before the final method is chosen and written down. Measure the floor and ceiling again on test seed 456 so the headline has its own reference lines.
- Results are always taken at the end of the 200 episodes, never from the best point along the way.
- Practice episodes are collected 10 at a time, with the same randomisation of friction, mass and gains as the test. Each of the 10 counts as one episode.
- Every learner's output starts at exactly zero, so before any learning it behaves the same as the original policy.
- Every run saves its final learner and scorekeeper, so later goals can reuse them without retraining.

Choosing the best setting without a human:
- pick the setting with the highest average success on the development test
- among all settings within one spread of that top average, pick the one with the fewest changes from the main experiment; if still tied, the higher average
- if no change in round 1 beats the main experiment by more than one spread, skip round 2 and confirm the main experiment
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
Status: passed on 2026-11-15 (zero test 60.4 percent against 61.4, cheat test 94.2 against 94.4). The fallback was not needed.

## Goal 4: build the learner

Achieve: a memory bank, a scorekeeper (several copies, layer normalisation), a corrector network, and a learning loop that studies the memory bank many times per new step. Build the learner so the number of outputs is a setting (2 for aim only, 3 for movement only, 5 for both), because Goal 6 needs all three.
Hand back: code and tests in `takehome/`.
Done when:
- a 10-episode trial run finishes with no errors and no invalid numbers
- the episode counter refuses to collect episode 201
- one command starts a full 200-episode experiment from a named policy and seed
- whenever both fill-up data and new data exist, each study batch is half of each; when only one kind exists (the zero-fill-up experiment, or briefing the scorekeeper before any new data), the batch comes entirely from the kind that exists
- the correction numbers are rescaled before the scorekeeper sees them, and the old policy's input scaling is reused
- these settings can be changed without editing code: number of fill-up episodes, how many times the memory bank is studied per new step, number of scorekeeper copies, and the aim caps. Every other item in Goals 6 and 7 is built later, when it is first needed, each behind its own setting and with its own short test. Do not build them now.
- the 10-episode trial run is a code test, not an experiment: it does not count as a setting tried
- the memory bank records, for every step: whether the episode ended by success or by running out of time (kept separate), the episode number and step number, whether it was a fill-up or learner episode, the old policy's own action, and the true camera error. The true camera error is stored for analysis and for the experiments that are explicitly allowed to use it, and is never shown to the learner otherwise
- a test-episode runner in `takehome/` records step-by-step traces (the running guess, the true error, the scorekeeper's rating, contact force), because the existing one only returns end-of-episode numbers

## Goal 5: run the main experiment (the one check-in with Griffin is after this)

Achieve: an honest measurement of what the aim corrector does in 200 episodes.
Hand back: learning curves and a results file for practice seeds 0 to 4, with exactly 40 fill-up episodes and 160 learner episodes each, tested as written under "Definitions".
Done when:
- all 5 seeds finished and every seed is shown, not only the average
- the final result is compared with the floor and ceiling from Goal 1
- the report says plainly whether it improved, stayed the same, or got worse
There is no target number. A flat or negative result reported honestly counts as done.
While Goal 5 runs, measure whether 1, 2 or 4 experiments side by side on the GPU finish sooner in total, and whether several seeds inside one process is faster still. Use real 200-episode runs for this so nothing is wasted.
After this goal, stop once and show Griffin:
- the Goal 1 floor and ceiling, the Goal 3 checks, and the learning curves
- how long one 200-episode run took, measured, and the side-by-side speed test result
- a time estimate table for everything that remains, one row per goal: number of runs, minutes per run, hours if run one at a time, and hours using the fastest side-by-side setup with helpers working in parallel. Give a likely figure and a worst case for the total, in hours of the laptop staying awake. Say plainly which part of the time helpers can shorten (writing code, tables, charts) and which they cannot (the single GPU)
- a cut list: which experiments to drop first if the total is too long, in order, and how many hours each saves
- how much of the Codex usage allowance has been used so far, if visible
Continue when he says go.

## Goal 6: comparisons at the same budget

Achieve: know how the aim corrector compares with the usual approach.
Hand back: one table and one chart showing: original policy (floor), no-camera-bias test (ceiling), aim corrector only (the Goal 5 result, reused), movement corrector only (small capped corrections to hand movements, same scorekeeper recipe, same 200 episodes), both together (one learner outputs the aim nudge and the movement correction, one scorekeeper, same 200 episodes), and a clearly labelled reference that cheats on information: a network trained on the same 200 practice episodes per seed, but using the true camera error as labels. It goes through the same episode counter, so it cheats on what it is told, not on how much practice it gets.
Done when: every learner used test seed 123 and practice seeds 0 to 4 (the floor and ceiling have no learning, so one run each), the combined learner starts from zero correction so it begins identical to the original policy, the cheating reference is labelled as such everywhere it appears, and the report says which of the three learners did best and what each part added.

## Goal 7: exploration round, then one confirmation run

Achieve: find the best settings through wide experimentation, and report a number that was not picked by peeking.
Explore one change at a time. Items about the aim part or the scorekeeper are one change from the main experiment (aim only). Items that need the movement part are one change from the Goal 6 "both together" learner; these are marked (movement). The 40 fill-up case is the main experiment itself, so reuse it.
Notes that settle the details:
- briefing the scorekeeper uses only that experiment's own fill-up episodes, never test episodes or old results; it cannot be combined with zero fill-up
- seeing the true camera error is off in the main experiment; when it is on, the final decider is not used
- the main scorekeeper is trained with an ordinary squared-error loss; "yes or no predictor" means swapping that for a classification-style loss
- the force penalty changes only what the learner is rewarded for; the success test never changes. Do not combine it with the yes or no loss
- sibling fill-up means the fill-up episodes are shared out among other `force_0.5mm` policies that succeed at least sometimes, run through the same wrapper; they count toward the 200 like any fill-up
- one after the other: split the learner episodes in half
- in round 2, skip any combination that cannot work together and log why
- limit: at most 45 settings in total across rounds 1 and 2, unless Griffin changes this at the check-in
The list:
- fill-up share: 0, 10, 20, 40, 80, 120 of the 200 episodes
- (movement) aim and movement together: joint (one learner, one scorekeeper), side by side (two learners, two scorekeepers, same time), one after the other (aim then movement, and movement then aim)
- scorekeeper briefed first or not
- scorekeeper sees true camera error in training or not
- several scorekeepers or one
- how many times the memory bank is studied per new step
- size caps on the aim nudge, and (movement) on the movement correction
- (movement) a final decider: each step, rate several options with the scorekeeper (no correction, aim only, movement only, both, a few small variations) and act on the best, versus acting on the learner output directly; use only a scorekeeper that does not need the true camera error
- hand over near the hole (old policy alone on the approach, learner active only near the hole)
- (movement) looser movement cap in contact, tighter in free space
- (movement) movement cap sized by the old policy's own uncertainty
- scorekeeper trained as a yes or no predictor of success
- looking several steps ahead when crediting moves
- a floor on the scorekeeper's ratings from real outcomes
- (movement) a penalty for pushing too hard, with peak force reported
- fill-up episodes from sibling policies
Search in three rounds, not every combination: (1) screening, each trick on by itself; (2) combining, only the few tricks that clearly beat the seed-to-seed spread; (3) confirmation, the single best setting on new seeds.
Hand back: one exploration table with every setting tried, all seeds shown, and the seed-to-seed spread next to each result.
Done when:
- all choosing used the development seeds under "Definitions" (test seed 123), and nothing touched the confirmation seeds before the choice was written down
- each experiment used its own 200 practice episodes
- the total number of settings tried is stated
- differences smaller than the seed-to-seed spread are marked as not a real difference
- the best setting is chosen by the fixed rule in the autopilot rules, with the choice and reason written down first
- the chosen setting is then run on the confirmation seeds (practice 100 to 104, test seed 456), and that result is reported as the headline even if it is lower than the exploration result
- the three Goal 6 learners at their plain settings are also run on the confirmation seeds, with the floor and ceiling measured on test seed 456, and the report says plainly that the final method had the benefit of tuning and the others did not, with the number of settings tried shown next to each method

## Goal 8: check the scorekeeper's honesty

Achieve: know whether the scorekeeper's confidence can be trusted.
Important detail: the scorekeeper's raw rating is not a chance of success, because wins that come later are counted for less and a time-out is not treated as a failure. So for this goal produce an honest "chance of success in this episode" reading: either convert the rating, or fit a small separate yes or no predictor on the same memory bank (no extra practice episodes) that also knows how many steps are left. Say in the report which you did and why. Also show the raw rating against outcomes for comparison.
Hand back: a chart of predicted chance of success against actual success, a list of confident-but-failed moments, and the step where confidence drops in failed episodes.
Done when: the chart is built only from test episodes, and the report says in one sentence whether the scorekeeper is overconfident, underconfident, or about right.

## Goal 9: failure cases

Achieve: a clear picture of how it still fails.
Hand back: counts of each failure type before and after (never touched the hole, stuck on the rim, ran out of time, corrected the wrong way), and a few videos per type showing the running guess next to the true error.
Done when: each failure type is defined by a rule that can be checked from the recorded episode, the rules are applied in a fixed order so every failed test episode lands in exactly one type, there is an "other" type for anything left over, and the counts add up. Videos come from a new script in `takehome/` that reuses the existing video code without editing it.

## Goal 10: check it is not a one-off

Achieve: know whether the result holds on other policies.
Hand back: the final method, unchanged, run on `force_0.5mm_s4`, on the position-only policy `pos_0.5mm_s1`, and on `force_0.5mm_s1`, which hovers and never inserts (0 percent with and without the camera error). Load policies through `peg_insertion/rollout.py` directly; do not run `scripts/evaluate.py`, because it writes into the existing results. Measure each policy's own floor and ceiling first. Position-only policies take 10 numbers instead of 16, so the wrapper must handle both.
Done when: all three are reported, including the ones expected to do poorly, with a sentence on why each behaved as it did.

## Goal 11: different camera problems

Achieve: know which kinds of camera error the corrector can and cannot fix.
Hand back: one table and one chart covering the seven problems in the plan (eight conditions, because the bigger offset has two sizes), each with floor, test-only, retrain, and ceiling. Floor is the old policy under that problem. Ceiling is the Goal 1 ceiling test (no constant offset and none of the added problem). Test-only uses the already trained final method from seeds 0 to 2 with nothing changed. For each problem, write down the exact formula and numbers used before running it. Where the offset is bigger, scale the cap on the running guess and the size of each nudge up by the same factor, in both the test-only and retrain runs, and log the values. For each problem also run the cheat test (true error fed in as the guess): it shows the best any aim corrector could do there, so a problem the aim idea cannot fix in principle is told apart from one the learner failed to learn. Build each problem by extending the environment class inside `takehome/`, since the height error and the late reading have to change the reading where it is made.
Done when: the new errors live only in `takehome/` code, each retrain used its own 200 episodes, and the height error is included even though it is expected to fail.

## Goal 12: write-up

Achieve: a report Griffin can send and defend.
Hand back: `TAKEHOME_REPORT.md` at the top of the project, in plain language with no em dashes, with the question, the idea, the method, all results with every seed, the charts, failure cases, every algorithm choice and its reason, the number of settings tried, every PLAN GAP entry, the limits, the budget assumption, how AI assistance was used, and one command to rebuild each figure.
Done when: every number in the report can be traced to a file in `results/takehome/`, and nothing is claimed that the results do not show.

## Goal 13: final checks

Achieve: confidence that the work is clean and repeatable.
Done when:
- all old and new tests pass
- `git diff main -- peg_insertion scripts tests results` shows no changes outside `results/takehome/`
- one fresh rerun of the final method on one confirmation practice seed lands within one spread of the reported result
- everything is committed and pushed to `origin takehome`
- a final summary is written for Griffin listing what to read first, every fallback that was used, and every question left open
