# Forecasting demand when the economy is part of the signal

**Question.** Industry demand for industrial trucks swings with the economy. If we forecast the macroeconomic drivers forward and feed them into a demand model, do we get a forecast the business can plan against — and is it better than assuming next year looks like last year?

**Setup.** A two-stage pipeline, as built for a graduate capstone with an industrial truck manufacturer. Stage 1 forecasts each driver (10-year rate, consumer confidence, CPI, retail trade volume) with SARIMAX, choosing the order by AIC rather than fixing one order for every series. Stage 2 regresses unit demand on trend, product class, and a subset of drivers. All 15 driver subsets are compared. Data is synthetic, generated to match the structure of the original problem; no client data is included. Validation is rolling-origin: three folds, each forecasting 12 months ahead using only prior history.

**Headline result.**

| Selection rule | Drivers chosen | In-sample R² | Backtest MAPE |
|---|---|---|---|
| Highest in-sample R² | all four | 0.972 | 16.48% |
| Lowest backtest error | 10-year rate only | 0.852 | **13.86%** |
| Seasonal naive benchmark | — | — | 14.38% |

The model with the best fit was the worst forecaster. Choosing on in-sample R² costs 2.6 points of MAPE against choosing on backtest error, and the R²-selected model does not even beat "same month last year." Only one specification beats the naive benchmark, and it is the simplest one.

**Why the fit is misleading here.** Comparing the two error columns in `results/model_comparison.csv` separates the two sources of error. With the drivers *known*, the four-driver model forecasts demand to 6.4% MAPE — the demand regression is genuinely good. With the drivers *forecast*, the same model degrades to 16.5%. Nearly all the damage comes from stage 1: errors in forecasting consumer confidence and CPI twelve months out swamp the information those variables add. A driver only earns its place in the model if it can be forecast about as accurately as the thing you are trying to predict, and that test is invisible to in-sample R².

**What this means for the forecast.** Ship the simple specification, report it against the seasonal naive baseline so planners know what the model is adding, and publish prediction intervals rather than point forecasts — the stage-1 uncertainty is large and currently hidden. If consumer confidence really does carry signal, the way to use it is a nowcast over a shorter horizon, not a twelve-month projection.

**Limits.** Synthetic data, so the absolute MAPEs describe this generated series rather than the real market. The relationship is linear and stable by construction, which flatters every model. Stage 1 forecasts each driver independently, ignoring the fact that rates, prices, and confidence move together — a VAR would respect that. Prediction intervals are not yet propagated through the two stages.

## Relationship to the original capstone

The capstone (team project, Jan–Jun 2025) built the same two-stage structure and searched every subset of drivers, keeping the specification with the highest R². This version changes four things:

1. Model selection is by rolling-origin backtest instead of in-sample R².
2. A seasonal naive benchmark is included, so "accurate" has a reference point.
3. SARIMAX orders are chosen per series by AIC rather than fixed at one order for all drivers.
4. The demand regression is scored both with forecast drivers and with known drivers, separating stage-1 from stage-2 error.

## Run it
```bash
pip install -r requirements.txt
python run.py            # full backtest
python run.py --quick
```
Outputs: `results/summary.json`, `results/model_comparison.csv`, `figures/forecast_backtest.png`.
