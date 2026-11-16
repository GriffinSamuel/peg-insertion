# Phase B design, fixed before new choosing runs

All settings use the original aim recipe, practice seeds0–2/test123500, extend any gain larger than both matched sample spreads to0–4. No test episode supplies targets, statistics or heat-map counts. The fourteen-slot aim limit includes the new camera-context aim baselines conservatively; fixed-condition plain movement is a fifteenth setting with no aim part. Code tests do not count as searched settings, but all practice episodes remain counted.

| Slot | Name | Single idea / required integrated method | Base |
|---|---|---|---|
|1|B_gate|Hold zero before observed contact; update only while observed depth has stalled.|plain aim|
|2|B_pullback|Subtract1% of current guess each control step.|plain aim|
|3|B_memory|Give actor and critic the mean and change over20 raw readings, alongside current readings and guess.|plain aim|
|4|B_labels|Add a fixed-weight1 mean-squared error toward labels derived from first-success camera XY, alongside the original Q objective.|plain aim|
|5|B_fast|Double aim nudge cap from0.1 to0.2mm after observed contact.|plain aim|
|6|B_constant|Use one random constant grid offset per fill episode; online initial guess remains zero.|plain aim|
|7|B_heat_normal|Across-episode success-rate grid with its required constant-offset fill and hottest starting aim.|plain aim|
|8|B_heat_one_sided|The same integrated grid method under one-sided camera bias.|B_aim_one_sided|
|9|B_heat_fixed|The same integrated grid method under fixed bias per run.|B_aim_fixed|
|10|B_aim_fixed|Plain aim under fixed camera bias.|own original-policy floor|
|11|B_aim_one_sided|Plain aim under one-sided camera bias, supplying the matched comparison.|own original-policy floor|
|12|B_within_grid|After height stalls near the opening, rule out failed grid cells and move toward the nearest unruled cell.|plain aim|
|13|B_qmap|Memory plus constant-offset fill, only if both isolated components HELPED; otherwise DID NOT RUN with reason.|plain aim|
|14|B_stack|Stack compatible changes that HELPED on five seeds; otherwise DID NOT RUN with reason.|plain aim|
|not aim|B_movement_fixed|Plain movement under fixed camera bias.|own original-policy floor|

Slot13 drawings, if eligible, evaluate Q over bounded candidate aim nudges from stored episodes after training; true camera error is used only for the overlay. They are not targets and cannot affect selection. Slot14 uses all compatible HELPED changes. Drift gate and a nonzero heat-map initial aim conflict, so they are not stacked. Context variants are not stacked into normal-camera candidates. A memory/constant-fill stack is not run if either isolated change fails the required five-seed verdict, because the user explicitly required only helped ideas to be combined.

Contact is detected from the original clipped force reading, norm >=5N. A stall is less than0.2mm mean camera-Z descent between the first and last five readings of the last ten readings. Histories contain20 observations and pad with the reset reading. No hidden depth or hole location is read by these controls. For memory, reuse the frozen policy normalizer for the mean and normalized difference of first/latest readings, and the existing guess scale for the two aim values.

The label is camera XY immediately after the first successful practice action. It includes clearance error and camera jitter: insertion does not reveal the exact hole centre. Assign it only to valid transitions in that successful practice episode; never use test success for labels. The actor predicts the absolute camera error normalized by the3mm guess cap, converts it to the bounded incremental correction, and receives an auxiliary squared-error target at that label, coefficient1, with failed episodes retaining the original reward objective. Do not tune coefficient or timing from results.

The offset grid is9x9 from-2 to+2mm per axis in0.5mm increments, inside the existing3mm guess cap. Constant fill chooses a cell uniformly once per episode using a folded key8000 independent of the simulator's original reset/action key schedule. Across-episode rates use Laplace counts (successes+1)/(attempts+2); each episode is credited once to its starting cell, and ties choose nearest zero, then fixed grid order. Recompute only from this experiment's counted replay. Report this online credit simplification as a limitation. The within-episode grid waits for observed camera height below1mm and a stall, marks a reached cell cold after five stalled steps, moves at most0.2mm/axis/step, and holds its last cell if all81 are ruled out. Its search logic uses no force reading; the frozen policy and otherwise unchanged learner retain their original observations. Mark this setting as using hand-written search and exclude it from Phase G.

Fixed-camera bias is drawn once per practice seed by JAX normal key fold8713, standard deviation0.75mm per axis. The same value stays inside the simulator for every practice, curve and final test episode in that run. One-sided bias is the componentwise absolute value of the original per-episode bias, exactly Goal11. Measure a500-test floor and oracle-corrected ceiling for every fixed seed and for the one-sided condition before those learners; those are test-only, not trained policies or choosing candidates. Normal-camera old floor/ceiling remain frozen references. Normal-camera old aim and movement seeds0–4 are reused, not rerun for favorable numbers.

Required integrated heat maps are tested regardless of the separate constant-fill verdict; this preserves the explicit all-three-camera-condition request. They are labelled integrated methods, and their gain over a plain base cannot identify which internal component helped. Optional stacks obey the separate-five-seed HELPED requirement. All final500 tests save local full traces; curve tests save summaries and per-episode outcomes only.

Before any new choosing result was inspected, clarify extension for the separate working-aim check: also extend a normal-camera aim candidate whose first three seeds beat61.4% by its own sample spread and whose paired success error improves, even if the older aim base has too wide a spread for a PROMISING verdict. Only the full five-seed result can satisfy the working definition. This adds seeds, not settings, and never changes a verdict threshold.

Implementation note recorded during review, without changing the completed within-grid setting: nearest-cell ties use the original fixed grid order (`argmin`). This reproducible tie rule can send the early path into one part of the grid rather than covering all directions. It remains unchanged; the later prescribed blind spiral is a separate Phase F comparison.
