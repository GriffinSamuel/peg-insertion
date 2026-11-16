# Goal 7 implementation design, fixed before screening

This is the lead's design for the next phase, prepared while Goal 6 runs. It does not authorize new seeds or change the fixed tests. Each new switch receives a contract test before use. The main implementation and its completed results remain the comparison.

Screen 24 new configurations, with practice seeds 0, 1, 2 and final 500 tests at seed 123. Reuse existing main and joint comparisons. The global count after screening is 28 including the four Goal 5/6 settings, below the stricter interpretation of the 45-setting cap.

| Change | Base | Values fixed for screening |
|---|---|---|
| Fill share | Aim | 0, 10, 20, 80, 120; reuse 40 |
| Coordination | Combined | Independent aim/movement actors and critic ensembles: simultaneous, aim then movement, movement then aim; reuse joint |
| Critic briefing | Aim | Four critic updates per valid own fill transition; actor stays frozen |
| Privileged critic | Aim | Append true XY bias divided by guess cap to critic input during training and labelled diagnostic recording only; never actor input |
| Critic ensemble | Aim | One copy; reuse four |
| Update ratio | Aim | One critic update per valid new step, actor every update; reuse four |
| Aim nudge caps | Aim | 0.05 mm and 0.2 mm per step; reuse 0.1 mm; guess cap stays 3 mm |
| Movement cap | Combined | 0.4 mm per step; reuse 0.2 mm |
| Final decider | Combined | Rate zero, aim only, movement only, both, half both, 1.5 times both clipped, half aim plus movement, aim plus half movement |
| Near-hole handover | Aim | Active when observed absolute height <15 mm and observed XY distance <10 mm; otherwise zero correction |
| Contact cap | Combined | Nominal cap times 0.5 below 5 N sensor force, times 2 at or above 5 N |
| Uncertainty cap | Combined | Each axis nominal cap times clip(2 times frozen-policy pre-tanh action standard deviation, 0.5, 2); no sampling |
| Classification loss | Aim | Sigmoid critic output with binary cross entropy and the same discounted target; raw Q still not a finite-episode probability |
| Multi-step credit | Aim | Three-step discounted return plus bootstrap, cut at success or time limit; bootstrap retained at time limit |
| Return floor | Aim | Bellman target at least the discounted observed return to the end of its own practice episode |
| Force penalty | Combined | Subtract 0.001 times max(force minus 15 N, 0) times 0.02 seconds per valid step; unchanged success test; report peak force |
| Sibling fill | Aim | Forty own counted fill episodes round-robin over force_0.5mm_s0, s3, s4: 20,10,10; their recorded original successes are nonzero |

The final-decider variations scale actor outputs, so every candidate is exactly zero at initialization. Independent actors have separate trunks, parameters, optimizer states, targets and critics. Sequential modes train the first component for 80 learner episodes and the second for 80; the inactive component is frozen, and the previously trained component continues acting. Neither mode shares practice across experiments.

Three-step targets and return floors are computed only inside the same counted practice episode, without crossing resets. Source sampling stays exactly half fill and half learner when both exist. Critic briefing uses its own fill only and the single-source exception. Force penalty and classification loss are incompatible; zero fill and briefing are incompatible; privileged critic and final decider are incompatible. The final-decider option is behavior and target-policy selection; actor gradients still improve the direct actor output.

Selection uses only development outcomes. For screening comparisons, use the same first three practice seeds from the relevant base, and conservatively require a gain larger than both methods' sample standard deviations. Treat differences below that threshold as unproven. Reused plain movement and combined methods are candidates as changes from the main aim method. If nothing beats the main method beyond that conservative spread, apply the written skip rule and confirm the main method. Otherwise, extend promising single changes to all five seeds, combine at most the best three compatible improvements (up to four pair/triple combinations), and choose by the written highest-mean/one-spread/simplest rule. Distinct output modes count as one change from main; each additional changed switch counts as one. No supervised cheating reference is eligible. Write the choice before any confirmation work.

These numeric choices are modest single alternatives or half/double scales chosen before results, not optimized values. Remaining defaults are the unchanged main recipe. All unsuccessful runs remain in the exploration table.
