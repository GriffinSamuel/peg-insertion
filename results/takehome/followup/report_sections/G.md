## Phase G: final check on fresh seeds

The choice was written before any fresh run (`G/choice.json`; advisor consultation 3 agreed). Rule, fixed in advance: highest development average, then fewest changes within one spread. The winner is the **plain movement corrector** (83.36% +- 2.99 points on development seeds, test seed 123). Nothing from Phases B to F beat it in the normal camera condition on five seeds, so the follow-up chose the method of the original study. The hidden-information scorekeeper (86.2%) was ineligible and not clearly better anyway (+2.9 points, inside the spread).

Fresh seeds 200 to 204, test seed 1001, 500 test episodes each, never used before:

| Reference or method | Result |
|---|---|
| Floor: original policy, normal camera error | 307 / 500 = 61.4% |
| Plain movement corrector (chosen), per seed | 74.2, 89.0, 85.2, 84.4, 88.0 |
| Plain movement corrector, mean +- spread | **84.2% +- 5.9 points** |
| Zero camera error reference | 475 / 500 = 95.0% |

Because the chosen method is the plain movement corrector, the "chosen versus plain movement" comparison is the same thing run once, with a difference of zero by construction. The improvement over the floor is 22.8 points. Development numbers (83.4%) were picked as best out of roughly 50 settings on one test seed, so they are somewhat optimistic by construction; only the fresh numbers are headline-grade. The original study's 80.2% was on a different set of seeds (confirmation seeds 100 to 104, test seed 456), so 80.2% and 84.2% are two draws of the same method and differ by less than the seed spread. **The follow-up did not beat the original result; it reproduced it.** A person reading this should know that the follow-up involved far more tuning (about 70 additional settings) than the original and found no real gain.

Not run: E_normal_move_then_aim (81.0% +- 15.8 on development seeds, one seed collapsed to 53%) as a secondary fresh row, because the locked final procedure runs only the chosen method.
