# Measuring a rollout you could not randomize

**Question.** A price increase goes out to some regions and not others. Cancellations rise afterwards. How much of that rise was the price change, and how much was seasonality and everything else happening at the same time?

**Setup.** A 40-region, 36-month panel of monthly cancellation rates, with 10 regions treated from month 25. Region baselines, a shared seasonal cycle, a common upward drift, and slow region-specific drift are all present, which is what makes naive before-and-after comparisons wrong. The true effect is set to +0.40 percentage points on a roughly 10% monthly churn rate, so each method can be scored against the answer. Run `--data` to point the same pipeline at a real panel.

**Three estimates, and what each one is for.**

| Method | Estimate | 95% interval | Notes |
|---|---|---|---|
| Two-way fixed effects DiD | +0.33 pp | [0.25, 0.41] | region and month fixed effects, SEs clustered by region |
| Synthetic control | +0.37 pp | — | pre-period fit RMSE 0.025 pp |
| True effect | +0.40 pp | — | known only because the data is simulated |

**Does the design hold up?**
- Event study: every pre-rollout coefficient sits near zero (largest 0.17 pp, intervals covering zero), so the parallel-trends assumption is not obviously violated before treatment.
- Synthetic control weights spread across 14 donor regions with no single region above 0.21, so the result does not hinge on one comparison unit.
- In-space placebos: the same synthetic control was run pretending each untreated region was treated. None of the 18 well-fitted placebos produced a gap as large as the real one, giving a pseudo p-value of 0.053.

**What this means.** Both designs recover an effect in the right neighbourhood, with DiD landing slightly low. For a decision, the honest statement is: churn rose by roughly a third of a percentage point in treated regions, the pre-period evidence supports the design, and a gap this large did not appear in any placebo region. The next step is commercial, not statistical: weigh incremental churn cost against the revenue gain per retained member.

**Limits.** Simulated panel with a clean single rollout date. Real rollouts are staggered, which makes two-way fixed effects biased (Goodman-Bacon); the staggered case needs a Callaway–Sant'Anna style estimator. Region-level data also hides who churned, so no heterogeneity by tenure or plan.

## Run it
```bash
pip install -r requirements.txt
python run.py
python run.py --data data/panel.csv   # columns: region, month, churn_rate, treated_from
```
Outputs: `results/summary.json`, `results/event_study.csv`, `figures/did_and_synthetic_control.png`.
