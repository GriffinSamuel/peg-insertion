# Phase D: original tricks on plain aim with fixed camera error

The revised plan explicitly requires this phase even though no normal-camera aim fix worked. The base is plain fixed-camera aim and its own same-seed Phase B results. Screen practice seeds0 to2 at test123, extend PROMISING results to0 to4, and keep200 practice episodes per experiment.

Run18 applicable tricks at the exact original Goal7 values: fill0,10,20,80,120; critic briefing; privileged critic; one critic; one update; aim nudge0.05mm and0.2mm; final decider; near-hole gate; classification critic; three-step credit; return floor; force penalty; sibling fill. Reuse fill40 as the base.

The final decider retains the four unique aim-only candidates from the original eight: zero, actor, half actor, clipped1.5times actor. Force penalty remains exactly0.001*max(force-15N,0)*0.02 in practice rewards only. Both are meaningful with aim-only control even though the original study tested them with both outputs. The near-hole gate has the original observed15mm height and10mm sideways thresholds. Hidden-information critics are clearly labelled and excluded from final selection. Sibling policies remain frozen and their own fresh fill episodes count toward the200.

List six inapplicable settings with reasons: all three two-component coordination modes, the movement-only cap, contact-dependent movement cap and uncertainty-dependent movement cap. These cannot change a controller with no movement output. The independent-pair modes belong in Phase E with both components.

Above-zero-error results here require extra seed789 physical checks under Griffin's revised consistent-camera rule. No score from those checks is used for choosing.
