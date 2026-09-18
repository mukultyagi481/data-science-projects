# Choosing a metric you can still defend in six months

**Question.** A team wants one primary metric for onboarding experiments. Candidates get proposed on intuition: watch hours, sessions, titles started. A metric is only useful if it is both sensitive enough to move in a test of realistic size and predictive of whether the member is still subscribed months later. Which candidate clears both bars?

**Setup.** A simulated sign-up cohort of 40,000 members with 14 days of activity and a 90-day churn outcome (44.8% churn over the window). Five candidate metrics are scored on sensitivity (smallest detectable relative lift at 80% power, 50,000 members per arm) and on predictive value for 90-day retention.

| Candidate (first 14 days) | Smallest detectable lift | AUC for 90-day retention | Retention gap, top vs. bottom quintile |
|---|---|---|---|
| Days active | 0.65% | 0.724 | 54.3 pp |
| Sessions | 1.37% | 0.662 | 38.3 pp |
| Titles started | 1.52% | 0.622 | 30.9 pp |
| Watch hours | 2.30% | 0.662 | 40.0 pp |
| Watched 3+ titles | 2.85% | 0.582 | 20.5 pp |

**Recommendation: days active in the first 14 days.** It is the most sensitive of the five by a factor of two, because a bounded count has a much smaller coefficient of variation than heavy-tailed watch hours, and it is also the best single predictor of 90-day retention. Watch hours look appealing because they sound closest to value, but the tail makes them the second-noisiest metric here — a test would need a 2.3% lift before it could see anything.

**Retention model.** Kaplan–Meier survival at 90 days: 61.0% for TV-first members, 53.0% for web, 50.3% for mobile. A Cox model on days active, log watch hours, device, and channel reaches a concordance of 0.686, and hazard ratios are in `results/cox_hazard_ratios.csv`.

**Caveats a reviewer should push on.** Sensitivity and prediction are not the whole test. A primary metric also has to be hard to game (days active can be inflated by nagging notifications, which is exactly the behaviour a guardrail metric should catch) and it has to stay stable as the product changes. Correlation with retention is not causation either: moving days active by a nudge does not guarantee the retention benefit follows, which is why the metric should be paired with a longer-horizon holdout.

**Limits.** Simulated data with a known engagement structure; real cohorts have seasonality, plan mix, and content-slate effects. Censoring is administrative only.

## Run it
```bash
pip install -r requirements.txt
python run.py
```
Outputs: `results/summary.json`, `results/sensitivity.csv`, `results/predictive_value.csv`, `results/cox_hazard_ratios.csv`, `figures/metric_design.png`.
