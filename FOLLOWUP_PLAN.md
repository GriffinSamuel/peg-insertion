# Follow-up plan: fix the aim corrector, retest the tricks, then try touch

Written 2026-11-15, after Goals 1 to 13 were finished and pushed (commit e9f399c). This plan is for a coding agent (Codex). It is extra work on top of a finished take-home. `AGENTS.md` and the "When to stop" and "Definitions" sections of `CODEX_GOALS.md` still apply in full.

## Where things stand

The finished result: a movement corrector raised success from 60.8 percent to 80.2 percent on the confirmation test. The aim corrector, the original idea, scored 62.8 percent, which is no real gain. None of the 24 extra tricks showed a gain, but most were tested on the aim corrector, which was not working, so they were never fairly tested on the method that does work.

## Ground rules for the follow-up

1. Nothing already reported changes. The 80.2 percent headline, every existing results file, and the existing text of `TAKEHOME_REPORT.md` stay exactly as they are.
2. New outputs go in `results/takehome/followup/`. New code goes in `takehome/`. The write-up goes in a new file at the top of the project, `TAKEHOME_FOLLOWUP_REPORT.md`, using the same plain language and no em dashes.
3. The budget rule is unchanged: 200 practice episodes per experiment, every learned-from episode counts, nothing is learned from test episodes.
4. Seeds, fixed now. Choosing: test seed 123, practice seeds 0 to 2. Anything that beats its base by more than one spread is extended to practice seeds 0 to 4 before it is believed. Final check: practice seeds 200 to 204 and test seed 1001, which have never been used. Nothing runs on those until the final choice is written down. Seeds 100 to 104, 456 and 789 are already used and must not be used for choosing or for the new final check.
5. Keep counting. Every new setting adds to the running count of settings tried, and the follow-up report states the total for the whole project.
6. A difference inside the seed-to-seed spread is reported as no real difference.
7. No planned check-in. Work through the phases in order and stop only under the existing "When to stop" rule. Write a phase report into `TAKEHOME_PROGRESS.md` and commit and push after every phase.
8. Limit: at most 90 new settings in total across all phases. If the limit is reached, finish the current phase, skip to Phase G, and say what was left out.

## The scoreboard (required for every setting)

Keep one running table for the whole follow-up at `results/takehome/followup/scoreboard.md`, with the same rows in `scoreboard.csv`. Add a row as soon as a setting finishes, and commit it with each phase. Every setting gets exactly one row, including ones that did badly or failed to run. Columns:

- phase, and a short name
- what it is, in one plain sentence
- what it is compared against (its base) and that base's result
- the result for each seed, as successes out of 500 and a percentage
- average and spread
- difference from the base, in points
- verdict, in capitals, using only these words: HELPED (beats the base by more than one spread, on five seeds), PROMISING (beats it on three seeds, not yet rerun on five), NO REAL DIFFERENCE (inside the spread), HURT (worse by more than one spread), UNSTABLE (at least one seed collapsed below half the base), DID NOT RUN (with the reason)
- for anything with an aim part: how far the guess was from the true camera error at the start and at the moment of success, in millimetres
- average steps to insert, and peak force
- practice episodes used
- two yes or no flags: uses a hand-written search routine, and uses hidden simulator information
- the camera condition (normal, one-sided, or fixed for the whole run)

At the top of `scoreboard.md`, keep a short plain-language summary that is rewritten after every phase: what has helped so far, what has not, and the current best learning method with its number. Add one chart per phase showing every setting's average and spread next to its base, with each seed as a dot.

The follow-up report must include the full scoreboard and must not leave out any row.

## Phase A: confirm why the aim corrector failed (no new episodes)

Using the saved final test traces from the five aim-only confirmation runs, verify or correct these observations, which came from a quick outside check:
- the running guess does not track the true camera error: the guess is about 1.0 to 1.6 mm from the truth at first contact and 1.6 to 2.4 mm at the end, against an average true error of 0.95 mm
- the guess ends closer to the truth in only 12 to 25 percent of episodes
- each seed drifts in its own fixed direction, whatever the true error is
- first contact happens around step 125 of 200
- the old policy succeeds about 90 percent of the time when the true error is under 0.5 mm and about 15 percent when it is over 1.5 mm
Also measure the guess error at the moment of success rather than at the end of the episode, and how well the true error can be predicted from one step of readings compared with the last 20 and 40 steps.
Hand back: one table and a few plain sentences saying which explanations the data supports.

## Phase B: fix the aim corrector

Test each explanation with one change, starting from the aim-only main experiment. Then stack the changes that helped.

1. Drift. Hold the guess at zero until first contact, and let it change only while insertion depth has stalled. Add a gentle pull back toward zero.
2. Memory. Let the corrector and the scorekeeper see the last 20 to 40 steps of readings, or a short summary of them.
3. Thin reward. In every successful practice episode, the true hole position becomes known when the peg goes in, so the camera error for that episode is known afterwards without reading any hidden simulator value. Train the corrector to predict that error directly from those labels, alongside the scorekeeper.
4. Speed. Allow a larger nudge per step after contact, so a correction fits in the steps that remain.
5. Heat map across episodes (Griffin's idea). Keep a grid over sideways aim offsets and record, from the robot's own practice episodes only, how often each offset led to success. Use the "hottest" offset as the starting aim. Fill the grid by trying one constant random offset per fill-up episode instead of tiny random nudges. Expected result, to be tested and reported either way: in the normal setting the camera error is drawn fresh each episode and averages zero, so the hottest cell should be near zero and this should not help; under the one-sided camera problem from Goal 11, where the direction is consistent, it should help. Also add a new camera condition built in `takehome/` only, "fixed for the whole run": one camera error is drawn per practice seed (same 0.75 mm spread) and stays the same for every practice and test episode of that run, like one robot with its own fixed miscalibration. Measure that condition's own floor and ceiling. Run the heat map on all three conditions, and run the plain aim corrector and the plain movement corrector on the fixed condition too for comparison.
6. Heat map within an episode. Keep a grid for the current episode only, marking offsets that were tried and stalled as cold, and move the aim to the nearest cell not yet ruled out. This is a search with memory and uses no force reading.
7. Heat map that depends on what the robot feels. This is what the scorekeeper already is: a rating for each correction given the current readings. Give it the memory from item 2 and the constant-offset fill-up from item 5, then draw its ratings over a grid of offsets for a few example episodes, next to the true error, so the report can show whether the hot spot sits where the hole really is.
8. Stacked. Combine whichever of 1 to 7 helped.

No method in this phase may use the hidden camera error from the simulator, in practice or in testing. It is used only afterwards, to measure how close the guesses were.

What "working" means, fixed now: on practice seeds 0 to 4 and test seed 123, aim-only success beats the 61.4 percent floor by more than one spread, and the guess is closer to the truth at the moment of success than at the start, on average.

Stopping rule for this phase: at most 14 aim variants. If none meets the definition, stop trying, write down plainly that the aim corrector could not be made to work within the budget and what the evidence says about why, skip Phase D, and continue.

### Added 2026-11-16 after the Phase B halt

1. Offset sweep, no learning. With zero camera error, run the old policy with a constant aim offset at each point of a 7 by 7 grid from -0.9 mm to +0.9 mm, 256 test episodes each at test seed 123. This measures directly where the old policy's sweet spot is. Draw it as a heat map. If the best cell is not the centre, the zero-error ceiling is not a true upper limit for a constant offset, and the report should say so and give the best cell's success rate as the real upper limit.
2. Finish the interrupted items: rerun fixed-camera plain aim seed 4 from scratch (keep the halted partial run on the scoreboard as its own row), run the heat map on the one-sided condition, and extend every PROMISING row to five seeds.
3. In the normal condition none of the aim fixes met the definition of working. Record that plainly. Because the aim corrector does work when the camera error is consistent, Phase D runs on the "fixed for the whole run" condition instead of being skipped, compared against plain aim on that condition. Phase E's aim-and-movement combinations run on both the normal and the fixed condition.

4. Mixed camera error (added 2026-11-16, evening). Real setups usually have both kinds of error at once: a part that stays the same for one robot (a crooked camera) and a part that changes on every attempt (where this particular piece landed, noise in this particular picture). Add a "mixed" condition in `takehome/` only: a fixed part drawn once per practice seed plus a fresh part drawn every episode. Run three splits with the same overall size as the normal error: mostly fixed (fixed 0.65 mm, per-episode 0.375 mm), even (0.53 mm each), and mostly per-episode (fixed 0.375 mm, per-episode 0.65 mm). For each, measure the floor and the zero-error ceiling, and run plain aim, the across-episode heat map, plain movement, and aim with movement together, on practice seeds 0 to 2 at test seed 123, extending anything PROMISING to five seeds. Report how much of the lost success each method recovers as the fixed share grows.

## Phase C: the 24 tricks on the movement corrector

Rerun every Goal 7 trick that makes sense for a movement-only learner as one change from the plain movement corrector (development result 83.4 percent, spread 3.0 points). For tricks that only make sense with an aim part, list them and say why they are skipped. Use the same values that were fixed in `takehome/goal7_design.md`, so nothing is tuned after the fact. Add one value the first round did not cover: a smaller movement cap of 0.1 mm.
Hand back: a table like the Goal 7 exploration table, every seed shown.

## Phase D: the tricks on the fixed aim corrector (only if Phase B worked)

Rerun the tricks that make sense for the aim part as one change from the working aim corrector. Same rules as Phase C.

## Phase E: combinations

1. If the aim corrector now works, rerun "both together" with the fixed aim part and the plain movement part, and the two one-after-the-other orders.
2. Take the tricks from Phases C and D that beat their base by more than one spread on five seeds, at most the best three per base, and try their combinations, skipping any that cannot work together.
3. If nothing beat its base, say so and skip this step.

## Phase F: finding the hole by touch

This phase is a comparison and a teacher. A hand-written search is not a learning method, so by itself it can never be the headline.

1. Check first, from saved traces, whether the wrist force and torque readings change in a usable way when the peg rests partly over the opening. If they do not, say so, explain what the simulation's contact model gives instead, and do only the blind spiral below.
2. Hand-written methods, no learning, each tested on 500 episodes at test seed 123, all acting only through the aim guess once depth has stalled after contact. The nudge cap may be raised so a search fits in the remaining steps; log the value.
   - Blind spiral: move the guess outward in a small widening spiral until depth increases.
   - Twist direction: use the wrist torque while pressing down to pick a direction, and move the guess that way until depth increases.
   - Four-sided probe: slide the guess one way along x until the readings show the peg's edge crossed the hole's edge, then the other way, and set the x guess to the midpoint. Repeat for y. Then let the old policy insert.
   For each, report success, how far the final guess is from the truth, steps taken, peak force, and how often it ran out of time.
3. Touch as a teacher, only if a method in step 2 comes within 10 points of the ceiling. One 200-episode experiment. Episodes 1 to 40 are fill-up: the touch method runs, every step goes into the memory bank as fill-up data, and the camera error it measured is that episode's label. Episodes 41 to 200 are learner episodes: a network that sees the last 20 to 40 steps of readings predicts the camera error and sets the aim guess; if depth still stalls, the touch method takes over and its measurement becomes that episode's label too. Each episode counts once. Also try 20 and 80 fill-up episodes. Compare the touch method alone, the network alone, and the network followed by the touch method, on success, steps to insert, and peak force.
4. If the touch-taught aim part works, try it together with the movement corrector.

## Phase G: final check and write-up

1. Choose the best learning method from Phases B to F by the same fixed rule as before (highest development average; among those within one spread, the fewest changes). Write the choice and reason down first. The chosen method may not use a hand-written search routine at any point, in practice or in testing, and may not use hidden simulator information. Everything from Phase F, including the touch-taught network, is reported in its own clearly labelled section as a comparison and cannot be chosen.
2. Run the chosen method on practice seeds 200 to 204 and test seed 1001. On the same seeds, also run the plain movement corrector, and measure the floor and ceiling on test seed 1001, so the comparison is fair.
3. Write `TAKEHOME_FOLLOWUP_REPORT.md`: what was asked, what Phase A found, what fixed or did not fix the aim corrector, the trick tables for both bases, the combinations, the touch results, the final check, the total number of settings tried in the whole project, and the limits. Say plainly whether the follow-up beat the original 80.2 percent result or not, and that the follow-up method had far more tuning than the original. If a touch method is part of the chosen method, say clearly that it starts from a hand-written search routine.
4. Add one short paragraph at the very end of `TAKEHOME_REPORT.md` pointing to the follow-up report. Change nothing else in it.
5. Run all tests, confirm the original code, policies and results are unchanged, commit and push.
