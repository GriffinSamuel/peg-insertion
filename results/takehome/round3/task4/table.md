| student | precision | weights | bytes | success % (mean of 5, test 1001, SD) | us per decision (CPU, batch 1, indicative) |
|---|---|---|---|---|---|
| full_h256 | fp32 | 70,915 | 283,660 | 84.5 (SD 6.9) | 46 |
| full_h256 | fp16 | 70,915 | 142,860 | 84.7 (SD 6.2) | 48 |
| full_h256 | int8 | 70,915 | 72,460 | 58.0 (SD 10.0) | 50 |
| half_h180 | fp32 | 36,183 | 144,732 | 84.3 (SD 6.7) | 39 |
| half_h180 | fp16 | 36,183 | 73,092 | 83.8 (SD 6.6) | 48 |
| half_h180 | int8 | 36,183 | 37,272 | 56.3 (SD 21.3) | 39 |
| quarter_h125 | fp32 | 18,253 | 73,012 | 84.5 (SD 6.4) | 45 |
| quarter_h125 | fp16 | 18,253 | 37,012 | 84.3 (SD 6.7) | 44 |
| quarter_h125 | int8 | 18,253 | 19,012 | 56.9 (SD 21.0) | 28 |
| teacher (old policy + corrector, two nets) | fp32 | 77,257 | 309,028 | recorded 84.2; reproduced 84.1 | 50 |
