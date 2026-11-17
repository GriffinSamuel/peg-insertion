# Brief 3: review the final choice and report conclusions before fresh-seed runs
Read only this brief and files it lists (results/takehome/followup/G/choice.json, scoreboard.csv). Ignore your definition's instruction to read ESCALATION.md, PLAN.md, PROGRESS.md. Answer in at most 40 lines: agree/disagree with choice, flaws in conclusions, anything to add to the report.

## Choice (written before any fresh-seed run; rule: highest dev mean, then fewest changes within one spread)
Chosen: original plain movement corrector (normal camera, 5 dev seeds, test seed 123): 83.36% +- 2.99. Other eligible normal-condition five-seed candidates: E_normal_move_then_aim 81.0 +- 15.8, E_normal_joint 74.5 +- 8.3, original aim 65.2 +- 7.7. Excluded: fixed/mixed-condition rows, hidden-information rows (C_privileged_critic 86.24 five seeds, uses hidden info), hand-search, Phase F. Floor 61.4%, zero-error ceiling ~94.5% (offset-sweep best 94.53 at centre).
So the follow-up chose the same method as the original 80.2% headline; fresh seeds 200-204, test 1001 will rerun it and plain movement is the same thing, so the fresh comparison is "movement vs floor/ceiling on new seeds" (and aim-based/E_normal_move_then_aim as a secondary row if cheap).

## Conclusions I plan to report
1. No normal-condition aim repair worked; none of 20 movement tricks HELPED (C); none of D's singles HELPED on 5 seeds (all NO REAL DIFFERENCE or UNSTABLE) in the fixed condition; so Phase E stacks skipped.
2. HELPED rows: fixed-camera aim with a 0.2 mm aim-step cap (D_aim_cap_200um, 97.56% vs 89.46% plain fixed aim), mixed-camera plain movement/aim, one-sided aim. Fixed-condition result exceeds the 94.5% zero-error constant-offset ceiling; advisor 1 attributes it to a sideways guess wiggle (~0.1 mm per step) rather than a better aim estimate; replays not run.
3. Joint aim+movement HURT in normal (74.5 vs 83.4); move-then-aim 81.0 +- 15.8 (one seed at 53%) no real difference. Fixed: joint 98.8 vs movement 93.8 NO REAL DIFFERENCE on 5 seeds.
4. Phase F: wrist wrench gives right half-plane 87% of the time (clipped 41%); blind spiral 62.2% vs floor 61.4% (no gain); teacher skipped; torque/four-sided not implemented (parked).
5. Follow-up did not beat the original 80.2%.
## Ask
Is choosing plain movement right, and is the fresh run worth running as written? Anything misleading in 1-5?
