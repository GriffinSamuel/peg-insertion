# Goal 7 development results

Every result is the final 500-episode test at seed 123 after 200 own practice episodes. Unused screening seeds are labelled not run. Spread is the sample standard deviation across practice seeds. Differences below the larger matched seed spread are unproven. Peak force is the mean of each episode's maximum force, in newtons.

| Setting | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seeds | Mean ± sample SD | Difference versus matched base | Per-seed mean peak force |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aim only | 365/500 (73.0%) | 278/500 (55.6%) | 315/500 (63.0%) | 305/500 (61.0%) | 366/500 (73.2%) | 5 | 65.2% ± 7.7% | difference below spread, unproven | s0:88.26 N; s1:81.7 N; s2:86.69 N; s3:84.08 N; s4:87.4 N |
| Movement only | 425/500 (85.0%) | 411/500 (82.2%) | 427/500 (85.4%) | 428/500 (85.6%) | 393/500 (78.6%) | 5 | 83.4% ± 3.0% | gain beyond spread | s0:98.43 N; s1:92.79 N; s2:94.98 N; s3:96.6 N; s4:91.65 N |
| Aim and movement | 421/500 (84.2%) | 378/500 (75.6%) | 363/500 (72.6%) | 407/500 (81.4%) | 396/500 (79.2%) | 5 | 78.6% ± 4.6% | gain beyond spread | s0:96.2 N; s1:94.8 N; s2:84.37 N; s3:95.82 N; s4:95.12 N |
| Supervised reference (cheats: true labels) | 325/500 (65.0%) | 349/500 (69.8%) | 328/500 (65.6%) | 327/500 (65.4%) | 325/500 (65.0%) | 5 | 66.2% ± 2.1% | difference below spread, unproven | s0:83.82 N; s1:88.71 N; s2:84.12 N; s3:85.72 N; s4:84.54 N |
| Fill 0 episodes | 306/500 (61.2%) | 313/500 (62.6%) | 316/500 (63.2%) | not run | not run | 3 | 62.3% ± 1.0% | difference below spread, unproven | s0:83.33 N; s1:86.48 N; s2:83.25 N |
| Fill 10 episodes | 319/500 (63.8%) | 303/500 (60.6%) | 350/500 (70.0%) | not run | not run | 3 | 64.8% ± 4.8% | difference below spread, unproven | s0:86.15 N; s1:83.32 N; s2:89.73 N |
| Fill 20 episodes | 362/500 (72.4%) | 308/500 (61.6%) | 307/500 (61.4%) | not run | not run | 3 | 65.1% ± 6.3% | difference below spread, unproven | s0:87.92 N; s1:80.13 N; s2:81.72 N |
| Fill 80 episodes | 243/500 (48.6%) | 332/500 (66.4%) | 375/500 (75.0%) | not run | not run | 3 | 63.3% ± 13.5% | difference below spread, unproven | s0:81.76 N; s1:86.52 N; s2:88.23 N |
| Fill 120 episodes | 306/500 (61.2%) | 326/500 (65.2%) | 82/500 (16.4%) | not run | not run | 3 | 47.6% ± 27.1% | difference below spread, unproven | s0:84.35 N; s1:84.95 N; s2:70.28 N |
| side by side | 222/500 (44.4%) | 331/500 (66.2%) | 390/500 (78.0%) | not run | not run | 3 | 62.9% ± 17.0% | difference below spread, unproven | s0:72.52 N; s1:86.32 N; s2:96.6 N |
| aim then move | 264/500 (52.8%) | 420/500 (84.0%) | 466/500 (93.2%) | not run | not run | 3 | 76.7% ± 21.2% | difference below spread, unproven | s0:82.8 N; s1:94.52 N; s2:100.6 N |
| move then aim | 373/500 (74.6%) | 358/500 (71.6%) | 440/500 (88.0%) | not run | not run | 3 | 78.1% ± 8.7% | difference below spread, unproven | s0:85.94 N; s1:95.78 N; s2:102.7 N |
| Critic briefing on own fill | 293/500 (58.6%) | 344/500 (68.8%) | 356/500 (71.2%) | not run | not run | 3 | 66.2% ± 6.7% | difference below spread, unproven | s0:83.66 N; s1:84.47 N; s2:87.75 N |
| True bias to critic during training | 337/500 (67.4%) | 324/500 (64.8%) | 324/500 (64.8%) | not run | not run | 3 | 65.7% ± 1.5% | difference below spread, unproven | s0:84.28 N; s1:83.77 N; s2:80.59 N |
| One critic | 351/500 (70.2%) | 332/500 (66.4%) | 323/500 (64.6%) | not run | not run | 3 | 67.1% ± 2.9% | difference below spread, unproven | s0:89.32 N; s1:88.36 N; s2:80.94 N |
| One update per new step | 348/500 (69.6%) | 292/500 (58.4%) | 63/500 (12.6%) | not run | not run | 3 | 46.9% ± 30.2% | difference below spread, unproven | s0:87.3 N; s1:81.83 N; s2:67.61 N |
| Aim cap 0.05 mm | 317/500 (63.4%) | 2/500 (0.4%) | 345/500 (69.0%) | not run | not run | 3 | 44.3% ± 38.1% | difference below spread, unproven | s0:82.26 N; s1:68.49 N; s2:88.39 N |
| Aim cap 0.2 mm | 383/500 (76.6%) | 322/500 (64.4%) | 315/500 (63.0%) | not run | not run | 3 | 68.0% ± 7.5% | difference below spread, unproven | s0:91.89 N; s1:84.19 N; s2:85.33 N |
| Movement cap 0.4 mm | 461/500 (92.2%) | 276/500 (55.2%) | 427/500 (85.4%) | not run | not run | 3 | 77.6% ± 19.7% | difference below spread, unproven | s0:100.4 N; s1:86.86 N; s2:97.53 N |
| Final critic decider | 255/500 (51.0%) | 392/500 (78.4%) | 326/500 (65.2%) | not run | not run | 3 | 64.9% ± 13.7% | difference below spread, unproven | s0:82.4 N; s1:91.03 N; s2:87.69 N |
| Hand over near hole | 318/500 (63.6%) | 327/500 (65.4%) | 321/500 (64.2%) | not run | not run | 3 | 64.4% ± 0.9% | difference below spread, unproven | s0:83.21 N; s1:84.59 N; s2:87.78 N |
| Contact-dependent movement cap | 360/500 (72.0%) | 395/500 (79.0%) | 234/500 (46.8%) | not run | not run | 3 | 65.9% ± 16.9% | difference below spread, unproven | s0:87.59 N; s1:90.89 N; s2:73.02 N |
| Old-policy uncertainty cap | 408/500 (81.6%) | 403/500 (80.6%) | 359/500 (71.8%) | not run | not run | 3 | 78.0% ± 5.4% | difference below spread, unproven | s0:87.63 N; s1:93.52 N; s2:90.46 N |
| Classification critic loss | 351/500 (70.2%) | 328/500 (65.6%) | 280/500 (56.0%) | not run | not run | 3 | 63.9% ± 7.2% | difference below spread, unproven | s0:87.87 N; s1:87 N; s2:82.47 N |
| Three-step credit | 360/500 (72.0%) | 304/500 (60.8%) | 387/500 (77.4%) | not run | not run | 3 | 70.1% ± 8.5% | difference below spread, unproven | s0:91.31 N; s1:86.8 N; s2:91.2 N |
| Observed return floor | 340/500 (68.0%) | 331/500 (66.2%) | 330/500 (66.0%) | not run | not run | 3 | 66.7% ± 1.1% | difference below spread, unproven | s0:88.41 N; s1:87.85 N; s2:81.57 N |
| Force penalty (peak force reported) | 359/500 (71.8%) | 399/500 (79.8%) | 254/500 (50.8%) | not run | not run | 3 | 67.5% ± 15.0% | difference below spread, unproven | s0:93.02 N; s1:95.39 N; s2:76.07 N |
| Sibling-policy fill-up | 314/500 (62.8%) | 318/500 (63.6%) | 290/500 (58.0%) | not run | not run | 3 | 61.5% ± 3.0% | difference below spread, unproven | s0:82.96 N; s1:84.92 N; s2:84 N |
