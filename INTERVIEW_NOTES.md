# Interview notes

Short answers to questions an interviewer is likely to ask about this project.
Numbers come from `results/summary.csv`.

**1. What did you build, in one sentence?**
A simulated Franka Panda robot arm that learns, with reinforcement learning, to push a square peg it is holding into a square hole, and an experiment measuring how success drops as the gap around the peg shrinks from 2 mm to 0.5 mm, with and without a wrist force sensor.

**2. Why peg insertion?**
It is the textbook "contact-rich" manipulation task and it maps directly onto real factory work: plugging in connectors, pressing in pins, assembling parts. NVIDIA's Isaac Lab ships a peg-insert task (Factory) for the same reason. It is also easy to make harder in a controlled way: you just shrink the gap.

**3. What does the robot actually control?**
Three numbers per step (every 20 ms): how far to move the peg tip in x, y and z, at most 2 mm per axis. An analytic inverse-kinematics solver turns that target into joint angles, and the arm's position servos track them. The gripper points straight down and does not rotate. This is the same control scheme as MuJoCo Playground's Franka "pick cube" task.

**4. What does the robot observe?**
Ten numbers: where the peg tip is relative to the hole (3), the tip velocity (3), its previous action (3), and how deep the tip is (1). The force version adds six more: force and torque at the wrist. The hole position it sees is deliberately wrong by a random amount each episode (0.75 mm standard deviation), like a slightly miscalibrated camera, plus 0.2 mm of per-step noise.

**5. Why add that error to the hole position?**
Without it, the task is too easy: a perfect position estimate plus a straight-down move always works (my scripted controller gets 100% at every clearance when it knows the true hole position). Real robots never know the hole position perfectly. The bias is chosen so that at 0.5 mm clearance it is bigger than the gap in about half the episodes, so the robot has to find the hole some other way.

**6. Why PPO?**
It is the standard on-policy algorithm for massively parallel GPU simulation: stable, simple, and it is what Brax and MuJoCo Playground provide and tune for this robot. I started from Playground's own tuned settings for the same arm and control scheme and changed as little as possible.

**7. How is the reward designed?**
A weighted sum of: getting close to the hole (reach), lining up over it in x-y (align), how deep the peg is inside the hole (insert), a one-time bonus on first success, and small penalties for jerky actions and very large forces. The episode does not end on success, and the insert reward keeps paying while the peg stays in, so the best strategy is to go in and stay in rather than hover above the hole.

**8. What went wrong along the way, and how did you fix it?**
Three things. (1) The arm sagged about 2 cm under gravity because the simulated servos are soft, so I turned on gravity compensation, which real Franka controllers do. (2) Training was unstable: one run reached 98% success and then collapsed to 0%. Lowering the learning rate from 5e-4 to 2e-4 fixed that case. (3) A 1 mm run still collapsed. I rolled out the failed policy and saw it landing on the edge of the hole and sliding off. The alignment reward was nearly flat for errors under 2 mm, so it never rewarded millimetre precision. Making that term sharp over 5 mm fixed it for every condition. All earlier results are archived, not hidden.

**9. Did force sensing help?**
TBD (fill from summary.csv: force vs position at 0.5 mm, mean over seeds, plus steps to success).

**10. Why can position-only control succeed at all when the hole estimate is wrong?**
TBD (explain: the policy also sees its own tip position and velocity, so when it pushes down and does not move, it "feels" the rim indirectly through proprioception and can search. Fill with the step-count difference.)

**11. What is domain randomization and why use it?**
Each of the 1024 parallel simulations gets slightly different physics: contact friction (plus or minus 30%), peg mass (plus or minus 20%) and arm servo stiffness (plus or minus 10%). A policy that works across all of them is less likely to depend on one exact simulator setting, which is the first step toward working on a real robot.

**12. How fast was training on a laptop GPU?**
TBD (steps/s and minutes per run from summary.csv). Using MuJoCo's Warp backend instead of the JAX backend was about 4x faster on this GPU.

**13. What MJX / tooling limitations did you hit?**
MJX only collides simple shapes (boxes, capsules, convex meshes), so the peg and hole are made of boxes. The latest Brax still calls a JAX function that newer JAX removed, so I added a small compatibility shim. GPU rendering (EGL) does not work under WSL2, so videos use slow software rendering (osmesa). Brax's evaluation sums metrics over an episode, so I wrote my own evaluation rollout for per-episode statistics.

**14. How do you know the results are not a simulator glitch?**
A hand-written controller that knows the true hole position inserts 100% of the time at all three clearances, and jams when started 1.5 mm off-centre at 1 mm and 0.5 mm, as physics says it should. The evaluation also checks every episode for a peg passing through a wall (the tip can never be further off-axis than the clearance while it is inside the hole); TBD such episodes were found.

**15. What would you do next?**
Port the task to NVIDIA Isaac Lab with the Newton physics engine and compare against its Factory peg-insert task; add a chamfer and a round peg; let the gripper rotate so the robot can tilt the peg (the classic search strategy); and try sim-to-real on a real arm with a wrist force/torque sensor.

**How was AI used?**
I directed the project and made the research question and design choices; the code, experiments and first drafts of the docs were produced with Claude Code (an AI coding assistant) following my written plan. Every decision and its reason is logged in DECISIONS.md.
