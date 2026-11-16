Original-policy floor: 307/500 (61.4%). No-camera-bias ceiling: 472/500 (94.4%). Both use test seed 123 with no learning.

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | Sample spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aim only | 365/500 | 278/500 | 315/500 | 305/500 | 366/500 | 65.16% | 7.74 points |
| Movement only | 425/500 | 411/500 | 427/500 | 428/500 | 393/500 | 83.36% | 2.99 points |
| Aim and movement | 421/500 | 378/500 | 363/500 | 407/500 | 396/500 | 78.60% | 4.60 points |
| Supervised reference (cheats: true labels) | 325/500 | 349/500 | 328/500 | 327/500 | 325/500 | 66.16% | 2.05 points |

Each learned result uses its own 200 practice episodes and 500 test-only episodes at seed 123. Aim-only results are reused from Goal 5.
