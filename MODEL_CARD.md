# Model card: Veritas mule-account detector (full_r3)

Format loosely follows Mitchell et al., *Model Cards for Model Reporting* (2019), with sections a model risk or AI governance reviewer would expect.

## Summary
| | |
|---|---|
| Purpose | Rank bank accounts each day by how likely they are to be acting as money mules (pass-through intermediaries), so a fixed analyst team works the most likely ones first |
| Output | A daily ranked list; the top K accounts become alerts for human review |
| Decision made by the model | None. It prioritizes alerts. Every action (account restriction, exit, SAR) is an analyst decision |
| Model type | LightGBM classifier with graph features and 3 rounds of neighbour-score propagation |
| Training data | IBM AMLworld HI-Small / HI-Medium (synthetic), labels from Sept 1–3 2022 |
| Status | Research prototype on synthetic data. Not validated on real data. Not for production use |

## Intended use
- **In scope:** alert prioritization for mule-account review in retail or small-business deposit accounts, as one input next to existing rules and KYC signals.
- **Out of scope:** automated account closure or blocking without review; credit, onboarding or any customer-eligibility decision; use as the sole transaction monitoring control; any population other than accounts with both inflow and outflow in the last 72 hours.

## Users and affected people
- **Users:** AML / fraud analysts and their managers.
- **Affected people:** account holders who get alerted. A false positive means a review and possibly friction (holds, information requests). Some real mules are themselves victims (recruited through job or romance scams), so case handling should allow for that.

## Performance (synthetic data, 3-seed means)
| Test | Alerts/day | Recall | Precision |
|---|---|---|---|
| HI-Medium Sept 8–10, held out, pre-registered | 300 | 0.52 | 47% |
| HI-Medium Sept 11–16, forward, frozen model | 300 | 0.43 | 49% |
| LI-Small (low-illicit), development | 300 | 0.44 | n/a |

Baselines, placebo and full results: [README.md](README.md) and `results/`.

## Known limits and risks
- **Synthetic data.** Laundering in AMLworld follows eight generator templates. The model may be partly learning the generator rather than mule behaviour. Real-world performance is unknown.
- **Unrealistic precision.** 47–49% comes from complete labels, templated laundering, clean mule behaviour, few legitimate pass-through accounts, and clustered positives. Expect it to fall sharply on real data. See README, "Production gaps".
- **Fixed-budget recall depends on volume.** Recall at a fixed budget fell from 0.52 to 0.43 when mule volume rose about 27%, while precision held at 47–49%. Monitor precision and recall-versus-achievable separately, so a staffing shortfall isn't mistaken for model drift.
- **Label lag not modeled.** Real confirmed labels arrive weeks to months late.
- **No explanations yet.** Alerts carry no reason codes.
- **Run-to-run noise.** About ±0.03 recall per seed from non-deterministic multithreaded aggregation.
- **Fairness.** The data has no customer attributes, so disparate impact can't be tested here. On real data, alert rates and false-positive rates should be compared across customer segments (age band, account tenure, geography, business vs personal) before use.

## Controls a deployment would need
1. **Validation** independent of the developer, on real labeled history, including a test with realistic label lag.
2. **Reason codes** on every alert (top features in plain language) so analysts can explain decisions and reviewers can audit them.
3. **Monitoring**, weekly:
   - alert hit rate (confirmed mules / alerts worked), against a set floor;
   - feature drift (population stability index on the top 20 features);
   - share of alerts from accounts under 30 days old, to catch over-reliance on one signal.
4. **Retraining** on a fixed cadence (start monthly, tighten if precision falls between retrains on real data), with each retrained model re-run against a frozen holdout before release.
5. **Fallback.** Keep the existing rule set running; if monitoring breaches a floor, revert alert allocation to rules.
6. **Change log and ownership.** Named model owner, version history, and a record of every threshold or budget change.

## Development integrity
Pre-registered tests (`prereg/`), shuffled-label placebo, a causality check on feature snapshots, and a record of every deviation (`results/FORWARD_TEST.md`). About 17 configurations were compared on HI-Small during development, so HI-Small results are optimistic.

## Version
full_r3, code in this repository. Contact: repository owner.
