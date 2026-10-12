# Resume bullets

Pick one. All numbers come from `results/summary.csv` (500 evaluation episodes per run, mean over 3-5 seeds).

1. Trained a simulated Franka Panda arm with GPU-parallel reinforcement learning (MuJoCo MJX + Brax PPO, 1024 parallel environments, ~68k steps/s on an RTX 5070 laptop GPU) to insert a peg with 0.5 mm clearance; adding a simulated wrist force sensor raised success from 17% to 69% (26 training runs, p = 0.044).

2. Built and evaluated a peg-in-hole insertion benchmark in MuJoCo Playground, measuring how learned success falls from 99% to 17% as clearance tightens from 2 mm to 0.5 mm, and showing that force/torque feedback recovers it to 69% by making training succeed in 4 of 5 seeds versus 1 of 5.

3. Diagnosed and fixed reinforcement-learning training collapse (learning-rate instability and a reward with no millimetre-scale signal) in a contact-rich robot insertion task, then ran a 26-run, multi-seed experiment on a laptop GPU (5.5 min per 20M-step run) with honest reporting of failures, built with AI assistance (Claude Code) under my direction.
