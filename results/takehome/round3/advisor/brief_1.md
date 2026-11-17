# Brief 1 (Tasks 2 and 3 design). Read ONLY this file, ROUND3_PLAN.md (Tasks 2, 3), takehome/round3_runner.py, takehome/followup_env.py. Ignore your definition's instruction to read ESCALATION.md, PLAN.md, PROGRESS.md. Answer in at most 40 lines.

## Facts so far
- Runner built; one run takes about 1 min (cache warm). Same seed rerun varies a lot: seed 0 plain movement gave 426/500 (R0) vs 360/500 (R1a smoke, same recipe). So single-seed differences below about 5 points are noise. Base = goal6/movement (83.36 +- 2.99, seeds 0-4); R0 wrapper check mean 80.1 (seeds 0-4).
- Obs (16 + 2 guess): depth idx 9 (m), force 10:13 (/20 N), torque 13:16 (/2 Nm), all clipped to [-1,1]. From practice_0100.npz (R1a seed0): depth>0.01 m in only 5.6% of steps; torque norm below 0.01 in 90% of steps, spikes at contact; torque saturates (|comp|>=1) in under 1% of steps.

## Task 2 design (shaping)
Done in Round3Runner.update: reward' = reward + g*Phi(next) - Phi(this), Phi(terminal)=0. Phi_depth = 0.20*min(depth/0.024,1), Phi_torque = -0.05*min(|torque|,1); success reward unchanged. Undiscounted episode sum bounded by about 0.75 < 1. Settings 2a depth, 2b torque, 2c both on movement; 2d best of those on plain aim (base ~64% seeds 0-2).
Q1: are these scales sensible, or should torque be larger given its low signal? Anything that breaks policy invariance that I missed (e.g. truncation handling, n_step)?

## Task 3.0 / 3.1 design (test only)
3.0: saved aim learners B_aim_fixed and D_aim_cap_200um (fixed camera error, seeds 0-2), 500 test eps at test seed 123, four ways: full; guess held at global mean of pre-contact guesses; learner until first contact (contact = force >= 5 N, existing contact_seen) then guess frozen; constant + i.i.d. random wiggle of 0.1 mm per step (uniform direction, applied after contact only, per-episode key).
3.1: old policy + circular wiggle (period 10 steps) added via movement correction after contact and while depth stalled (existing stalled()), amplitude 0.05/0.1/0.2 mm. Counted as 3 settings (no learning).
Q2: is applying the wiggle only after contact a fair test of "the wiggle is what helps"? Should (4) also wiggle before contact?

## Task 3.2 design (learned search)
Env SearchEnv(FollowupEnv, outputs=3 internally) maps a learner action p in [-1,1]^4 to a movement: after first contact (zero before), width=0.3mm*relu(p0), lean=(p1,p2)*0.2mm, extra push down=0.2mm*relu(p3); circular wiggle width*[cos,sin](2*pi*k/8), k=steps since contact; sum clipped per axis to the 0.2 mm cap. relu makes p=0 (initial actor output) = exactly the old policy. New SearchLearner subclass (own actor size 4 or 7, action_scale=1). Same 40 fill + 160 learner episodes, same counter.
- 3a: new p every step (4 outputs).
- 3b: p decided every 10 steps and held. To keep Q-learning consistent I plan macro-steps: train only on rows with step%10==0, reward = discounted sum over the 10 steps, discount g^10, next_obs 10 steps later (compact the replay before train_block). 
- 3c: better of 3a/3b plus plain movement added (7 outputs). Problem: if 3b wins, the plain movement part cannot be per-step inside a macro-step.
Q3: is the macro-step plan sound? For 3c, if 3b is better, should I (a) hold the plain movement too, (b) use 3a's structure anyway, or (c) park 3c?
Q4: is a circular wiggle with fixed period the right "search" family, or should phase speed also be learned (5th output)?
Q5: anything in Tasks 2-3 that looks like it would silently use hidden information or break the budget counter?
