# Can you beat "assume no change" on a real, volatile series?

**Question.** Forecasts get judged against each other and almost never against the trivial alternative. Take a real input-cost series — monthly US natural gas prices, which feed operating expense and supplier forecasts — and ask a narrow question: which models actually beat the naive benchmarks, at which horizons, and is the difference distinguishable from luck?

**Setup.** Real data: 354 monthly observations, January 1997 to July 2026, from public mirrors of EIA (Henry Hub price) and BLS (CPI-U), cached locally on first run. Prices are deflated to constant dollars before modelling. Four models — random walk, seasonal naive, Holt-Winters ETS, and SARIMA with the order chosen by AIC on each training window — are scored over 78 expanding-window origins at horizons of 1, 3, 6, and 12 months, using MASE (scale-free) and MAPE. Each model is then compared against the random walk with a paired Diebold-Mariano test.

**What I found.**

| Horizon | Random walk | ETS | SARIMA | Seasonal naive |
|---|---|---|---|---|
| 1 month | **0.760** | 0.808 | 0.812 | 2.030 |
| 3 months | **1.356** | 1.442 | 1.434 | 2.247 |
| 6 months | **1.851** | 1.885 | 1.908 | 2.267 |
| 12 months | **2.281** | 2.337 | 2.352 | 2.281 |

*MASE; lower is better. Full table with MAPE in `results/accuracy_by_horizon.csv`.*

The random walk wins at every horizon. Not one model beats it by a statistically detectable margin — the Diebold-Mariano p-values against the random walk range from 0.06 to 0.23, so the differences are consistent with noise. Seasonal naive is *significantly worse* at one and three months (p < 0.005): gas prices carry real seasonality in demand, but the price level itself is dominated by a random-walk component, and forecasting last winter's price is worse than forecasting last month's.

**Why this is the useful answer.** A twelve-month-ahead MAPE of about 45% is the honest planning number for this series, and no amount of model selection changed it much. The conclusions that follow are practical: quote input-cost forecasts as ranges rather than point estimates, don't let a model's in-sample fit justify replacing a naive benchmark, and put effort into hedging and scenario planning rather than into squeezing a better point forecast out of a near-random-walk price. A model that cannot beat the benchmark is still a result worth reporting — it tells the business where not to spend.

**One detail worth knowing.** At a twelve-month horizon the seasonal naive and random walk forecasts are identical by construction — "same month last year" and "the last observation" are the same value when the forecast origin sits twelve months back. That is why their rows match exactly and the test returns nothing at h=12, not a bug.

**Limits.** One series, so this says nothing about whether structural models beat naive forecasts for demand volumes, which are usually more predictable than prices. Absolute-error loss only; an asymmetric cost of over- versus under-forecasting would change the ranking. The SARIMA grid is small and fixed rather than a full search, and no exogenous drivers (storage levels, weather, production) are included — the obvious next step, and the one most likely to actually beat the benchmark.

## Run it
```bash
pip install -r requirements.txt
python run.py            # ~3 minutes, downloads data on first run
python run.py --quick
```
Outputs: `results/summary.json`, `results/accuracy_by_horizon.csv`, `results/diebold_mariano.csv`, `results/backtest_detail.csv`, `figures/forecast_comparison.png`.
