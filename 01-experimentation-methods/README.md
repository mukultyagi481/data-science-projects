# How much faster can an A/B test finish?

**Question.** Engagement metrics are skewed and noisy, so tests on small changes need a lot of traffic. We already know how each member behaved before the test started. How much of that noise can we remove, and what does peeking at the results every day cost us?

**Setup.** Simulated weekly watch hours for a two-arm test. Watch time is lognormal, so a small group of heavy members drives the mean, and a member's pre-period watch time correlates 0.8 with their in-experiment watch time. Because the data is simulated, the true effect is known, so bias and interval coverage can be checked directly rather than assumed. 400 replications per sample size, true lift 2%.

**What I found.**

| Estimator | Users needed for 80% power | Change vs. baseline |
|---|---|---|
| Difference in means | 72,157 | — |
| CUPED | 53,421 | −26% |
| OLS with covariate | 53,421 | −26% |

- All three estimators were effectively unbiased (bias under 0.001 on a true effect of 0.075) and 95% intervals covered the true effect 95–96% of the time, so the variance reduction is not bought with distorted inference.
- CUPED and regression adjustment gave identical answers to three decimals, which is expected: with one covariate they are algebraically near-equivalent. Worth knowing before arguing about which to implement.
- Peeking is the bigger problem. With no true effect at all, a fixed-horizon test hit p < 0.05 5.1% of the time, as it should. Monitoring daily for 14 days and stopping at the first significant result hit 22.3%. Roughly one in four "wins" from a daily-monitored test with no real effect.

**What this means for a test plan.** Use pre-period data as a covariate wherever members have history; it buys about a quarter of the sample size for free. Do not let anyone stop a test early on an unadjusted look. If early stopping matters to the team, use a sequential design (alpha spending or always-valid confidence sequences) rather than repeated fixed-horizon tests.

**Limits.** Simulated data, one covariate, one metric family. Members with no pre-period history gain nothing here, so the real-world saving depends on what share of the test population is new. A follow-up would use a predicted-engagement covariate for new members.

## Run it
```bash
pip install -r requirements.txt
python run.py            # full study
python run.py --quick    # fast check
```
Outputs: `results/power_and_coverage.csv`, `results/summary.json`, `figures/power_curves.png`.
