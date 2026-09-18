"""
Project 5 — Forecasting demand when the economy is part of the signal

Two-stage pipeline, as built for a graduate capstone with an industrial truck
manufacturer:

  stage 1: forecast each macroeconomic driver (rates, CPI, confidence, retail
           volume) forward with SARIMAX
  stage 2: regress industry unit demand on those drivers plus trend and product
           class, then feed the stage-1 forecasts in to project demand forward

The capstone selected the stage-2 model by picking the highest in-sample
R-squared out of every possible subset of drivers. That is the part worth
redoing: R-squared always rises as you add variables, and searching every
subset for the best one guarantees an optimistic answer. This version keeps the
two-stage structure and swaps the selection rule for rolling-origin
backtesting, so every number below is out-of-sample.

All data is synthetic, generated to match the structure of the original
problem. No client data is included.

Usage:
    python run.py                 # full backtest (~2 min)
    python run.py --quick
"""

import argparse
import json
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2026)
OUT = os.path.dirname(os.path.abspath(__file__))

ECON_COLS = ["interest_rate_10_yr", "consumer_confidence_index",
             "consumer_price_index", "retail_trade_vol_millions"]
CLASSES = ["class_1", "class_2", "class_3"]
HORIZON = 12          # months forecast ahead
N_FOLDS = 3           # rolling-origin folds


# ----------------------------------------------------------------------------
# Synthetic data shaped like the original problem
# ----------------------------------------------------------------------------
def simulate_data(months=132, rng=RNG):
    idx = pd.date_range("2014-01-01", periods=months, freq="MS")
    t = np.arange(months)

    def ar1(sd, phi=0.85):
        e = rng.normal(0, sd, months)
        out = np.zeros(months)
        for i in range(1, months):
            out[i] = phi * out[i - 1] + e[i]
        return out

    econ = pd.DataFrame(index=idx)
    econ["interest_rate_10_yr"] = 2.4 + 0.012 * t + ar1(0.18)
    econ["consumer_confidence_index"] = 95 + 6 * np.sin(2 * np.pi * t / 12) + ar1(3.0)
    econ["consumer_price_index"] = 230 * np.exp(0.0021 * t) + ar1(1.1)
    econ["retail_trade_vol_millions"] = 420_000 + 900 * t + 18_000 * np.sin(2 * np.pi * t / 12) + ar1(6_000)

    # True demand relationship: trend, class level, seasonality, and economics
    betas = {"interest_rate_10_yr": -380.0, "consumer_confidence_index": 42.0,
             "consumer_price_index": -6.5, "retail_trade_vol_millions": 0.0035}
    rows = []
    for ci, cls in enumerate(CLASSES):
        level = [5200, 3600, 2100][ci]
        signal = (level + 7.5 * t
                  + 260 * np.sin(2 * np.pi * t / 12 + ci * 0.4)
                  + sum(econ[c].values * b for c, b in betas.items()))
        signal = signal - signal.mean() + level
        units = signal + rng.normal(0, 0.05 * level, months)
        rows.append(pd.DataFrame({"order_month": idx, "truck_class": cls,
                                  "industry_units": np.maximum(units, 0)}))
    demand = pd.concat(rows, ignore_index=True)
    return econ, demand


# ----------------------------------------------------------------------------
# Stage 1: forecast the drivers, choosing the SARIMAX order by AIC
# ----------------------------------------------------------------------------
ORDER_GRID = [((0, 1, 1), (0, 1, 1, 12)),
              ((1, 1, 0), (0, 1, 1, 12)),
              ((1, 1, 1), (1, 1, 0, 12)),
              ((2, 1, 1), (0, 1, 1, 12))]


def forecast_driver(series, horizon, grid=ORDER_GRID):
    """Fit each candidate order on the training series, keep the lowest AIC."""
    logged = np.log1p(series)
    best = None
    for order, seasonal in grid:
        try:
            fit = SARIMAX(logged, order=order, seasonal_order=seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
        except Exception:
            continue
        if best is None or fit.aic < best[0]:
            best = (fit.aic, order, seasonal, fit)
    if best is None:
        raise RuntimeError("no SARIMAX order converged")
    aic, order, seasonal, fit = best
    pred = fit.get_forecast(steps=horizon).predicted_mean
    return np.expm1(pred), {"order": order, "seasonal_order": seasonal, "aic": round(aic, 1)}


# ----------------------------------------------------------------------------
# Stage 2: demand regression
# ----------------------------------------------------------------------------
def design_matrix(frame, econ_subset):
    x = pd.DataFrame(index=frame.index)
    x["month_ind"] = frame["month_ind"]
    for cls in CLASSES[1:]:
        x[f"is_{cls}"] = (frame.truck_class == cls).astype(float)
    for c in econ_subset:
        x[c] = frame[c].values
    return sm.add_constant(x)


def fit_demand(frame, econ_subset):
    y = frame.industry_units
    return sm.OLS(y, design_matrix(frame, econ_subset)).fit()


def mape(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


# ----------------------------------------------------------------------------
# Rolling-origin backtest of the whole two-stage pipeline
# ----------------------------------------------------------------------------
def build_panel(econ, demand):
    panel = demand.merge(econ, left_on="order_month", right_index=True, how="left")
    months = np.sort(panel.order_month.unique())
    month_ind = {m: i + 1 for i, m in enumerate(months)}
    panel["month_ind"] = panel.order_month.map(month_ind)
    return panel.sort_values(["month_ind", "truck_class"]).reset_index(drop=True)


def backtest(econ, demand, subsets, n_folds=N_FOLDS, horizon=HORIZON):
    """For each fold: forecast drivers from history only, then predict demand.

    Three error numbers per subset:
      pipeline   - drivers forecast, as the model would actually be used
      oracle     - drivers known, isolating the demand regression's own error
      naive      - same month last year, the benchmark any model must beat
    """
    panel = build_panel(econ, demand)
    months = np.sort(panel.order_month.unique())
    results = {s: {"pipeline": [], "oracle": []} for s in subsets}
    naive_errors, driver_notes = [], {}

    for fold in range(n_folds):
        cut = len(months) - (n_folds - fold) * horizon
        train_months, test_months = months[:cut], months[cut:cut + horizon]
        train = panel[panel.order_month.isin(train_months)]
        test = panel[panel.order_month.isin(test_months)].copy()

        # stage 1 on training history only
        forecast_econ = {}
        for c in ECON_COLS:
            series = econ.loc[econ.index.isin(train_months), c].asfreq("MS")
            pred, note = forecast_driver(series, horizon)
            forecast_econ[c] = pred.values
            driver_notes[f"fold{fold + 1}:{c}"] = note
        fc = pd.DataFrame(forecast_econ, index=pd.DatetimeIndex(test_months))

        test_fc = test.drop(columns=ECON_COLS).merge(
            fc, left_on="order_month", right_index=True, how="left")

        # seasonal naive benchmark: same month, previous year
        test_idx = pd.DatetimeIndex(test_months)
        prev_months = test_idx - pd.DateOffset(years=1)
        prev = panel[panel.order_month.isin(prev_months)]
        key = prev.set_index([prev.order_month + pd.DateOffset(years=1), "truck_class"]).industry_units
        naive_pred = [key.get((m, c), np.nan) for m, c in zip(test.order_month, test.truck_class)]
        mask = ~pd.isna(naive_pred)
        naive_errors.append(mape(test.industry_units[mask], np.array(naive_pred)[mask]))

        for subset in subsets:
            model = fit_demand(train, list(subset))
            results[subset]["pipeline"].append(
                mape(test.industry_units, model.predict(design_matrix(test_fc, list(subset)))))
            results[subset]["oracle"].append(
                mape(test.industry_units, model.predict(design_matrix(test, list(subset)))))

    rows = []
    for subset in subsets:
        model_full = fit_demand(panel, list(subset))
        rows.append({
            "drivers": ", ".join(subset) if subset else "(none)",
            "n_drivers": len(subset),
            "in_sample_r2": round(float(model_full.rsquared), 4),
            "backtest_mape_pipeline": round(float(np.mean(results[subset]["pipeline"])), 2),
            "backtest_mape_oracle": round(float(np.mean(results[subset]["oracle"])), 2),
        })
    return pd.DataFrame(rows), float(np.mean(naive_errors)), driver_notes


def make_figure(table, naive_mape):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=160)

    axes[0].scatter(table.in_sample_r2, table.backtest_mape_pipeline,
                    s=36, color="#0e8c6a")
    best_r2 = table.sort_values("in_sample_r2", ascending=False).iloc[0]
    best_oos = table.sort_values("backtest_mape_pipeline").iloc[0]
    for row, label, colour in [(best_r2, "picked by R²", "#c0392b"),
                               (best_oos, "picked by backtest", "#101820")]:
        axes[0].scatter([row.in_sample_r2], [row.backtest_mape_pipeline], s=110,
                        facecolors="none", edgecolors=colour, linewidths=2)
        axes[0].annotate(label, (row.in_sample_r2, row.backtest_mape_pipeline),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=9, color=colour)
    axes[0].set_xlabel("in-sample R²")
    axes[0].set_ylabel("backtest MAPE (%)")
    axes[0].set_title("A better fit is not a better forecast")

    order = table.sort_values("backtest_mape_pipeline").head(6)
    y = np.arange(len(order))
    axes[1].barh(y - 0.2, order.backtest_mape_pipeline, height=0.38,
                 color="#0e8c6a", label="drivers forecast")
    axes[1].barh(y + 0.2, order.backtest_mape_oracle, height=0.38,
                 color="#8a96a3", label="drivers known")
    axes[1].axvline(naive_mape, color="#c0392b", linestyle="--", linewidth=1.3)
    axes[1].annotate("seasonal naive", (naive_mape, len(order) - 0.6),
                     color="#c0392b", fontsize=9, rotation=90, va="top")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([d[:38] for d in order.drivers], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("backtest MAPE (%)")
    axes[1].set_title("Best six driver sets, out of sample")
    axes[1].legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures/forecast_backtest.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    econ, demand = simulate_data()
    subsets = [c for r in range(1, len(ECON_COLS) + 1)
               for c in combinations(ECON_COLS, r)]
    if args.quick:
        subsets = [s for s in subsets if len(s) <= 2]

    table, naive_mape, driver_notes = backtest(
        econ, demand, subsets, n_folds=2 if args.quick else N_FOLDS)
    table.to_csv(os.path.join(OUT, "results/model_comparison.csv"), index=False)
    make_figure(table, naive_mape)

    by_r2 = table.sort_values(["in_sample_r2"], ascending=False).iloc[0]
    by_oos = table.sort_values(["backtest_mape_pipeline"]).iloc[0]
    summary = {
        "data": "synthetic, structured like the original capstone problem",
        "months": int(len(econ)),
        "folds": 2 if args.quick else N_FOLDS,
        "horizon_months": HORIZON,
        "candidate_models": int(len(subsets)),
        "seasonal_naive_mape": round(naive_mape, 2),
        "selected_by_in_sample_r2": {
            "drivers": by_r2.drivers, "in_sample_r2": by_r2.in_sample_r2,
            "backtest_mape": by_r2.backtest_mape_pipeline},
        "selected_by_backtest": {
            "drivers": by_oos.drivers, "in_sample_r2": by_oos.in_sample_r2,
            "backtest_mape": by_oos.backtest_mape_pipeline},
        "cost_of_selecting_on_r2_pct_mape": round(
            float(by_r2.backtest_mape_pipeline - by_oos.backtest_mape_pipeline), 2),
        "driver_forecast_orders": driver_notes,
    }
    with open(os.path.join(OUT, "results/summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(table.sort_values("backtest_mape_pipeline").to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "driver_forecast_orders"}, indent=2))


if __name__ == "__main__":
    main()
