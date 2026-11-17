# Brief 1: why do fixed-camera aim runs beat the best constant offset?
Read only this brief and files it lists. Ignore your definition's instruction to read ESCALATION.md, PLAN.md, PROGRESS.md. Answer in at most 40 lines: mechanism verdict, what extra check (no new practice episodes) would confirm it, how to word it in the report.

## Question
Offset sweep (results/takehome/followup/B/offset_sweep/, 7x7 grid -0.9..+0.9 mm, zero camera error, old policy, 256 eps, seed 123): best constant aim offset gets 94.53% (ties centre). Yet fixed-camera aim learners (scoreboard.csv, D rows, 3 seeds) score 95-98% (D_return_floor 97.7, D_near_hole 97.1, D_fill_120 96.3), while their aim guess is not closer to the true camera error at success than at the start (aim_start 0.91 mm, aim_success 0.85-1.99 mm). Is it a time-varying guess (within-episode) that a constant offset cannot do?

## Evidence (traces: results/takehome/followup/D/<setting>/seed_<s>/test_0200_traces.npz, fields next_guess[T,N,2], true_bias, force, depth, success, valid; test seed 123, 500 eps; fixed bias per seed)
- Guess changes on ~100% of steps; total guess path per episode 25 mm in plain B_aim_fixed (final guess sits ~2 mm from the truth, 85-93% success; weaker than D runs).
- D runs, success episodes with contact (force>=5 N): guess error vs truth at first contact 0.2-0.8 mm (better than start 0.9), at success 0.6-1.8 mm (worse than at contact). Guess is near the truth before contact then moves away after.
- Post-contact guess path 2.5-3.9 mm but net displacement 0.8-0.9 mm: tortuosity median 3.1 (return_floor), 4.5 (near_hole); i.e. back-and-forth wiggle, not a drift. Failures wiggle similarly (path 3.2-4.2, tortuosity 4.9-6.9).
- Correlation of guess speed with depth stall: -0.03 (near_hole), +0.28 (return_floor): weak.
- Guess std across episodes at contact is only 0.06-0.16 mm per axis, so it is nearly a constant per seed before contact.

## Tried
Only the above summaries. No replay with a constant offset equal to the learned pre-contact guess.

## Need
Your verdict on the mechanism (pre-contact better-than-zero aim vs post-contact wiggle acting as a search/dither), and the cheapest decisive no-practice check (e.g. replay test episodes with the guess frozen at its contact value, or with the wiggle removed; is that allowed as a test-only run?).
