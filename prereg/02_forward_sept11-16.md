# Forward test, HI-Medium Sept 11–16: pre-registration
Written 2026-11-20, before any model is scored on Sept 11–16. Looked at so far: rows and laundering counts per day only.

## Why only HI-Medium
HI-Small has 46–396 rows a day after Sept 10, essentially the laundering tail, so it can't be used. HI-Medium has normal volume through Sept 16 (0.8–3.0M rows/day), then a tail from Sept 17 on, which is excluded.

## Model
full_r3, frozen. Same features, thresholds and hyperparameters, seeds 0–2. **Same training as step 4:** labels from Sept 1–3, early stop on Sept 6–7. No retraining on later days. This tests how a model trained once holds up 8–13 days later.

Code: the step-4 logic moved into one script with the test days as a parameter. Before scoring Sept 11–16, the script must reproduce step 4 on Sept 8–10: HI-Small 0.58–0.62 and HI-Medium 0.52–0.54 at 900 alerts. If it doesn't, stop and fix the port, not the model.

## Data
HI-Medium rows with ts < 2022-09-17. Daily 72h snapshots for days 1–16. Test days are 11–16 (6 days).

## Positives
Accounts that both sent and received a laundering transaction with ts in [Sept 11, Sept 17).

## Metric
- **Primary:** recall at 300 alerts/day (1,800 over 6 days), already-alerted accounts skipped (step-4 rule).
- Also reported: 180/day; scaled 723 and 1,204/day (step-4 ratio of 4.014, fixed now); precision; base model, best single feature, rule and random at the same budgets.
- Bar for "held up": primary recall ≥ 0.45, i.e. within about 0.08 of the step-4 fixed-budget result (0.53). Below that, report it as decay.

## Reporting
One run, reported as-is, whatever the result.
