Advisor answer 1 (summary written by the main session; see transcript for full text)
Q1 keep scales 0.20/0.05; shaping code correct if terminated mask matches critic (it does: train_block uses batch['terminated']) and next_obs[t]==obs[t+1] (true: both come from the same step output). Compare against R0 wrapper runs, not only 83.36; do not read 3-seed differences under ~8 points.
Q2 post-contact wiggle is fair; add control 2b (guess held at mean END-of-episode guess); say whether (4) is jitter or random walk, keep within step cap; use paired same-key comparisons.
Q3 macro-step plan sound: hold in env, noise once per block, compact a copy only, log update count (updates proportional to macro rows); log fraction of p0<=0, p3<=0; 3c = option (b): 3a structure with 7 outputs, decided now.
Q4 keep fixed circle, use period 10 in both 3.1 and 3.2.
Q5 no hidden info found; assert oracle=False; 3.0/3.1 test-only; take the 3.0 constant from saved practice files.
