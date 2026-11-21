# Forward test result (pre-registered in prereg/02_forward_sept11-16.md)

## Port check (required before scoring)
| | Required band | Result |
|---|---|---|
| HI-Small, recall at 900 alerts | 0.58–0.62 | 0.588 (pass) |
| HI-Medium, recall at 900 alerts | 0.52–0.54 | **0.516 (just under)** |

**Deviation, recorded:** the HI-Medium port came in 0.004 under the band. Before scoring Sept 11–16:
- Snapshot features from the original script (`step4/feat.py`) and this repo (`src/features.py`) were compared on HI-Small, days 2, 8 and 10: identical (max relative difference < 1e-4, same accounts).
- The rest of the model code is copied line for line.
- Re-running the original, unchanged script on HI-Small moved individual seeds by up to 0.027 (0.577 → 0.604). So the band, set from the earlier assumption of ±0.005 noise, was too narrow.

The forward test went ahead on that basis. A second HI-Medium run of the port (`rerun/`, run after the forward test, same code and seeds) gave **0.529** (seeds 0.510 / 0.541 / 0.535), inside the band and equal to the original script. The 0.516 was run-to-run noise.

## Result (3 seeds; Sept 11–16, 2,064 positives, 498,090 test accounts)
| Detector | 180/day | **300/day (primary)** | 723/day | 1,204/day |
|---|---|---|---|---|
| full_r3 | 0.352 | **0.427** (0.421–0.434), precision 49% | 0.526 | 0.576, precision 16% |
| base model | 0.242 | 0.288 | 0.355 | 0.402 |
| best single feature (up_pt_n) | 0.069 | 0.095 | 0.134 | 0.163 |
| hand rule | 0.016 | 0.023 | 0.052 | 0.082 |
| random | 0.002 | 0.004 | 0.009 | 0.013 |
| placebo (full_r3, shuffled labels) | 0.000 | 0.002 | 0.004 | 0.007 |

**Against the pre-registered bar (≥ 0.45 at 300/day): not met (0.427). Reported as decay.**

Context, not a re-scoring: precision at 300/day was 49%, against 47% on Sept 8–10. Mules per day rose from 271 to 344, and the frozen model got no new labels after Sept 7.
