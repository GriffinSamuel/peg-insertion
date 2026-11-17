# Round 3 report: four ideas aimed at known weaknesses

Written 2026-11-17. Plain language; terms are defined when first used.

## Short answer

**Nothing beat plain movement.** Plain movement means: the old policy stays frozen and a small learned network adds a capped correction (0.2 mm per step) to the hand movement. The final method is unchanged from round 2. On fresh seeds it scored **83.3%** (five practice runs, SD 4.9 points) on test seed 2002, against a floor (old policy alone) of 60.2% and a no-camera-error reference of 93.4%. Round 2 and later rounds had far more tuning than the original study; this round added 13 more settings (107 in the whole project), all tested on the same development test seed 123, so numbers there are optimistic. The fresh-seed run is the honest one.

Some words. A *success rate* is the share of 500 test episodes where the peg went in. *Spread* is the standard deviation of success across practice seeds. *Noise*: the simulator is not exactly repeatable. The same recipe on the same seed 0 gave 85.2% in one run and 72.0% in another, and two identical recipes (the old base and this round's rerun, "R0") differ by 3.3 points on five seeds. So "NO REAL DIFFERENCE" here means "none detected", roughly under 10 points on three seeds and under 6 on five.

## The four weaknesses and what was tried

| Weakness | Idea | Result |
|---|---|---|
| 1. Learning dips and runs end anywhere from 72 to 89% | Task 1: use only practice data to keep the best saved version (1a pick at end, 1b roll back during training, 1c both) | Same final success: 84.1% (1b, 1c) vs 83.4% (old base) vs 80.1% (rerun). 1a: 81.1% on three seeds. Lowest curve dip after episode 80 on seeds 0 to 2: 72.9% plain, 68.5 (1a), 67.1 (1b), 74.6 (1c). So no stabilising effect was shown, and since 1a does not change training, shallower dips cannot be credited to rollback. Rollbacks fired 0 to 5 times per run. |
| 2. Success signal has no slope | Task 2: add a "getting warmer" reward from depth (2a), wrist torque (2b), or both (2c), in a form that cannot change which behaviour is best (potential-based shaping); 2d: torque on the aim-only corrector | 2a 77.2%, 2b 83.7%, 2c 81.7% vs 84.2% base: none helped. 2a (-7 points) is unresolved, not shown harmless. 2d: 64.1% vs 63.9% plain aim: no difference. Likely reasons: depth is nonzero in only about 6% of steps and torque is near zero 90% of the time, so the extra reward is rare. |
| 3. Sideways wiggling seems to help by accident | Task 3.0 replay test, 3.1 hand-set wiggle, 3.2 learned search settings | See below. The wiggle explanation is not supported. Learned search was worse than plain movement. |
| 4. Nothing connects to making models small and fast | Task 4: shrink to one network, lower precision | Shrinking to one network a quarter the size lost nothing at 32 or 16 bit. See below. |

Task 5 (combining) was skipped: nothing earned HELPED. Task 6's choice rule picked plain movement (written to `results/takehome/round3/final_choice.json` before any fresh seed ran).

## Task 3 in detail

**3.0, what does the aim learner's wiggle do?** Test only, on saved fixed-camera-error aim learners (runs 0 to 2), 500 test episodes each:

| learner | full | one constant guess (pre-contact mean) | constant (end-of-episode mean) | learner until first contact, then frozen | constant + 0.1 mm jitter after contact |
|---|---|---|---|---|---|
| B_aim_fixed | 90.0% | 63.4% | 45.1% | 88.9% | 63.5% |
| D_aim_cap_200um | 99.2% | 34.7% | 6.0% | 97.1% | 34.4% |

Freezing the guess at first touch keeps almost all of the gain, so moving the guess while touching is not what helps; the gain is made before contact. The constant-guess arms are poor, so the jitter arm cannot rescue them and is uninformative rather than confirming. Caveats: the frozen old policy still reacts after contact (this only rules out the corrector's wiggle); this was tested on aim correctors, while the final method is a movement corrector; the 1 to 2 point loss from freezing may be a small real after-contact effect. "The learned guess is a path, not one number" is a guess; an arm with a constant equal to the guess at first contact was not run.

**3.1, hand-set circular wiggle** (hand-written routine, no learning; old policy plus a wiggle once touching and stalled, test seed 123): 0.05 mm 60.8%, 0.1 mm 62.2%, 0.2 mm 61.4%, against 61.4% for no wiggle. The wiggle was active in 98% of episodes (17% of steps), so it was a real test, and it did nothing.

**3.2, learned search settings** (a designed routine turns learner outputs into wiggle width, sideways lean and extra push; zero before contact; same 200 episodes): 3a every step 72.5%, 3b held for 10 steps 69.6%, 3c with plain movement on top 70.9% (decided in advance to build on 3a). All worse than plain movement (HURT; robust for 3a and 3b, narrow for 3c because of its larger spread). Learning the search beat a constant wiggle (about 61%) but restricting the learner to a designed routine cost 12 to 15 points against free movement. These methods contain a hand-written search routine.

## Task 4: shrinking the improved policy

Teachers: round 2 plain-movement learners, seeds 200 to 204, test seed 1001 (each reproduced its recorded result to within noise: 84.1% reproduced vs 84.2% recorded on average). Each student is one network (16 readings in, 3 movement numbers out) trained only to copy the teacher's noise-free action on that run's own recorded practice steps; no new practice episodes. 16-bit and 8-bit are **simulated by rounding the weights**, not a real low-precision runtime. Times are one decision on the laptop CPU, batch of one, **indicative only** (the GPU was busy with other jobs).

| student | precision | weights | bytes | success % (mean of 5, SD) | microseconds per decision |
|---|---|---|---|---|---|
| full_h256 | fp32 | 70,915 | 283,660 | 84.5 (SD 6.9) | 46 |
| full_h256 | fp16 | 70,915 | 142,860 | 84.7 (SD 6.2) | 48 |
| full_h256 | int8 | 70,915 | 72,460 | 58.0 (SD 10.0) | 50 |
| half_h180 | fp32 | 36,183 | 144,732 | 84.3 (SD 6.7) | 39 |
| half_h180 | fp16 | 36,183 | 73,092 | 83.8 (SD 6.6) | 48 |
| half_h180 | int8 | 36,183 | 37,272 | 56.3 (SD 21.3) | 39 |
| quarter_h125 | fp32 | 18,253 | 73,012 | 84.5 (SD 6.4) | 45 |
| quarter_h125 | fp16 | 18,253 | 37,012 | 84.3 (SD 6.7) | 44 |
| quarter_h125 | int8 | 18,253 | 19,012 | 56.9 (SD 21.0) | 28 |
| teacher (old policy + corrector, two nets) | fp32 | 77,257 | 309,028 | recorded 84.2; reproduced 84.1 | 50 |

Chart: `results/takehome/round3/task4/success_vs_size.png` (success against number of weights, one line per precision, teacher and old-policy-alone as reference lines). Findings: one network a quarter the size (18,253 weights vs 77,257 for the two-network system) matched the teacher at 32-bit and 16-bit (about 84%), at 4 times fewer weights and 8 times fewer bytes at 16-bit. Plain per-layer 8-bit rounding collapsed to about 57%, a crude method; smarter quantisation was not tried. Speed was about the same (tens of microseconds, dominated by call overhead). The best student was not more than 5 points below its teacher, so the extra-practice variant was not needed. This is a measurement on round 2 learners and has no bearing on the method choice.

## Final check

Plain movement, practice seeds 300 to 304, test seed 2002 (never used before), the Round 3 runner: 450, 399, 405, 394, 434 of 500, mean **83.3%** (SD 4.9). Floor 301/500 (60.2%), no-camera-error reference 467/500 (93.4%). Compared with round 2's fresh check (84.2%, floor 61.4, reference 95.0) this is the same within noise. Because plain movement was chosen, it was run once.

## Scoreboard

Full table: `results/takehome/round3/scoreboard.md` and `.csv` (one row per setting, including every flag). Base for movement rows is the existing plain movement on the same seeds. Counts: 13 settings this round (3 + 4 + 3 + 3 hand-wiggle sizes), 107 in total. The rerun of plain movement through this round's runner (R0) was a check, not a setting.

## Limits

- Same-seed repeatability is poor (13 points on one seed). Three-seed spreads (1.7 to 12.5) are unreliable.
- Test seed 123 has been used for choosing across 107 settings, so development numbers are optimistic; only the fresh run is clean.
- Task 3.0 used fixed-camera-error aim learners; the jitter arm was i.i.d. random offsets, not a random walk.
- Task 2 scores were chosen once (0.20 depth, 0.05 torque) and not tuned.
- Task 4 8-bit is simulated with naive per-tensor rounding; timings are indicative.
- Two advisor consultations were used (design of Tasks 2 and 3; final choice and conclusions; the Task 4 design was not separately consulted because it is a pure measurement, logged in `TAKEHOME_DECISIONS.md`).
