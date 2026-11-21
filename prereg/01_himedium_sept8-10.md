# HI-Medium fresh test: pre-registration (written 2026-11-20 16:40 MT, before any HI-Medium model run)

Model: full_r3 from step2b, code unchanged (feat.py, chain.py, chain4.py `full` mode, 3 seeds 0-2).
No feature, threshold, hyperparameter or K changes. One run. Result reported as-is.

## Primary test (same protocol as HI-Small)
- Data: HI-Medium_Trans.csv, rows with ts < 2022-09-11 (same cut as Small).
- Train labels Sept 1-3, early-stop Sept 6-7, test Sept 8-10. Positive = account that both sent and received laundering Sept 8-10.
- Metric: recall at 540 / 900 alerts (180 / 300 per day), already-alerted skipped. Pass bar: 0.60 at 900.

## Budget scaling (secondary, decided now)
HI-Medium is a bigger institution. Also report recall at a scaled budget:
K_scaled = K x (mean daily test-day candidate count, Medium / Small), where candidates = accounts with both inflow and outflow in the 72h snapshot.
The fixed absolute budget (180/300/day) is the primary number; the scaled one is the same alert rate as Small.

## Sanity check before primary
Re-run HI-Small full in this environment; must reproduce 0.60-0.62 at 900 alerts.

## Not done
No looking at HI-Medium labels beyond counts until the run finishes. Sept 11-16 untouched (reserved for a later forward test).
