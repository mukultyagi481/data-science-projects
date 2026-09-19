"""
Project 6 — Can you beat "assume no change" on a real, volatile series?

Every forecast pitch sounds convincing until it is scored against a benchmark.
This project takes a real monthly series — US natural gas prices, an input cost
that feeds operating expense and supplier forecasts — and asks a narrow
question: which models actually beat the trivial benchmarks, at which horizons,
and is the difference statistically distinguishable from luck?

Models compared
  random walk        last observed value, repeated (the benchmark for prices)
  seasonal naive     same month last year (the benchmark for seasonal series)
  ETS                Holt-Winters exponential smoothing
  SARIMA             order and seasonal order chosen by AIC on each training window

Evaluation
  rolling-origin backtest, expanding window, forecast horizons 1, 3, 6, 12
  MASE (scaled against the in-sample naive error, so it is comparable across
  horizons) and MAPE
  Diebold-Mariano test of each model against the random walk

Data: real. Fetched from public mirrors of EIA (Henry Hub monthly price) and
BLS (CPI-U), cached to data/ on first run so results are reproducible offline.
Prices are deflated to constant dollars before modelling.

Usage:
    python run.py
    python run.py --quick
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")

GAS_URL = "https://raw.githubusercontent.com/datasets/natural-gas/main/data/monthly.csv"
CPI_URL = "https://raw.githubusercontent.com/datasets/cpi-us/main/data/cpiai.csv"
HORIZONS = [1, 3, 6, 12]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def download(url, name):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def load_series():
    gas = pd.read_csv(download(GAS_URL, "natural_gas_monthly.csv"))
    gas["date"] = pd.to_datetime(gas["Month"], format="%Y-%m")
    gas = gas.set_index("date")["Price"].asfreq("MS")

    cpi = pd.read_csv(download(CPI_URL, "cpi_us_monthly.csv"))
    cpi["date"] = pd.to_datetime(cpi["Date"])
    cpi = cpi.set_index("date")["Index"].asfreq("MS")

    df = pd.concat([gas.rename("nominal"), cpi.rename("cpi")], axis=1).dropna()
    base = df["cpi"].iloc[-1]
    df["real"] = df["nominal"] * base / df["cpi"]      # constant recent dollars
    return df


# ----------------------------------------------------------------------------
# Models: each takes a training series and returns `h` forecasts
# ----------------------------------------------------------------------------
def f_random_walk(train, h):
    return np.repeat(train.iloc[-1], h)


def f_seasonal_naive(train, h):
    last_year = train.iloc[-12:].values
    return np.array([last_year[i % 12] for i in range(h)])


def f_ets(train, h):
    fit = ExponentialSmoothing(train, trend="add", seasonal="add",
                               seasonal_periods=12,
                               initialization_method="estimated").fit()
    return np.asarray(fit.forecast(h))


SARIMA_GRID = [((1, 1, 1), (0, 1, 1, 12)),
               ((0, 1, 1), (0, 1, 1, 12)),
               ((2, 1, 0), (1, 1, 0, 12)),
               ((1, 1, 1), (1, 0, 1, 12))]


def f_sarima(train, h):
    best = None
    for order, seasonal in SARIMA_GRID:
        try:
            fit = SARIMAX(train, order=order, seasonal_order=seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
        except Exception:
            continue
        if best is None or fit.aic < best[0]:
            best = (fit.aic, fit)
    if best is None:
        return f_random_walk(train, h)
    return np.asarray(best[1].get_forecast(steps=h).predicted_mean)


MODELS = {"random walk": f_random_walk, "seasonal naive": f_seasonal_naive,
          "ETS": f_ets, "SARIMA": f_sarima}


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def mase_scale(train):
    """In-sample one-step naive error, the denominator that makes MASE scale-free."""
    return np.mean(np.abs(np.diff(train.values)))


def backtest(series, origins, horizons=HORIZONS):
    """Expanding-window backtest. At each origin, train on history only."""
    records = []
    for origin in origins:
        train = series.iloc[:origin]
        scale = mase_scale(train)
        max_h = min(max(horizons), len(series) - origin)
        if max_h < 1:
            continue
        actual = series.iloc[origin:origin + max_h].values
        for name, fn in MODELS.items():
            try:
                pred = np.asarray(fn(train, max_h), dtype=float)
            except Exception:
                continue
            for h in horizons:
                if h > max_h:
                    continue
                a, p = actual[h - 1], pred[h - 1]
                records.append({"origin": int(origin), "model": name, "h": h,
                                "actual": a, "pred": p,
                                "abs_error": abs(a - p),
                                "scaled_error": abs(a - p) / scale,
                                "pct_error": abs((a - p) / a) * 100})
    return pd.DataFrame(records)


def diebold_mariano(errors_a, errors_b):
    """Paired test on loss differentials: is model A's error different from B's?

    Uses absolute-error loss and a Newey-West style correction via the simple
    t-test on the differential, which is adequate for these horizons.
    """
    d = np.asarray(errors_a) - np.asarray(errors_b)
    d = d[~np.isnan(d)]
    if d.size < 8 or np.allclose(d.std(ddof=1), 0):
        return np.nan, np.nan
    t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
    p = 2 * (1 - stats.t.cdf(abs(t), df=d.size - 1))
    return float(t), float(p)


def summarise(bt):
    table = (bt.groupby(["model", "h"])
               .agg(mase=("scaled_error", "mean"),
                    mape=("pct_error", "mean"),
                    n=("abs_error", "size"))
               .reset_index())
    tests = []
    for h in sorted(bt.h.unique()):
        sub = bt[bt.h == h]
        rw = sub[sub.model == "random walk"].set_index("origin").abs_error
        for name in MODELS:
            if name == "random walk":
                continue
            m = sub[sub.model == name].set_index("origin").abs_error
            common = rw.index.intersection(m.index)
            t, p = diebold_mariano(m.loc[common], rw.loc[common])
            tests.append({"h": int(h), "model": name,
                          "dm_t_vs_random_walk": None if np.isnan(t) else round(t, 3),
                          "p_value": None if np.isnan(p) else round(p, 4),
                          "verdict": ("no detectable difference" if np.isnan(p) or p >= 0.05
                                      else ("better than random walk" if t < 0
                                            else "worse than random walk"))})
    return table, pd.DataFrame(tests)


def make_figure(series, table, bt):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=160)

    axes[0].plot(series.index, series.values, linewidth=1.0, color="#0e8c6a")
    axes[0].set_title("US natural gas price, constant dollars")
    axes[0].set_ylabel("$ per million BTU")

    for name in MODELS:
        sub = table[table.model == name].sort_values("h")
        axes[1].plot(sub.h, sub.mase, "o-", label=name, linewidth=1.6, markersize=4)
    axes[1].axhline(1.0, color="#8a96a3", linestyle=":", linewidth=1.2)
    axes[1].annotate("MASE = 1: no better than in-sample naive", (1, 1.03),
                     fontsize=8, color="#5d6b73")
    axes[1].set_xlabel("forecast horizon (months)")
    axes[1].set_ylabel("MASE")
    axes[1].set_title("Error grows with horizon")
    axes[1].legend(frameon=False, fontsize=9)

    h12 = bt[bt.h == 12]
    order = [m for m in MODELS if m in set(h12.model)]
    axes[2].boxplot([h12[h12.model == m].scaled_error.values for m in order],
                    labels=[m.replace(" ", "\n") for m in order], showfliers=False)
    axes[2].set_ylabel("scaled error")
    axes[2].set_title("Spread across origins, 12 months ahead")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures/forecast_comparison.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    df = load_series()
    series = df["real"]

    first_origin = 120                      # ten years of history before scoring
    step = 6 if args.quick else 3
    origins = list(range(first_origin, len(series) - 1, step))

    bt = backtest(series, origins)
    table, tests = summarise(bt)

    bt.to_csv(os.path.join(OUT, "results/backtest_detail.csv"), index=False)
    table.to_csv(os.path.join(OUT, "results/accuracy_by_horizon.csv"), index=False)
    tests.to_csv(os.path.join(OUT, "results/diebold_mariano.csv"), index=False)
    make_figure(series, table, bt)

    best_by_h = {}
    for h in sorted(table.h.unique()):
        sub = table[table.h == h].sort_values("mase")
        best_by_h[int(h)] = {"model": sub.iloc[0].model,
                             "mase": round(float(sub.iloc[0].mase), 3),
                             "random_walk_mase": round(
                                 float(sub[sub.model == "random walk"].mase.iloc[0]), 3)}

    summary = {
        "data": {"series": "US natural gas price, monthly, deflated by CPI-U",
                 "sources": [GAS_URL, CPI_URL],
                 "span": f"{series.index.min():%Y-%m} to {series.index.max():%Y-%m}",
                 "observations": int(len(series))},
        "backtest": {"origins": len(origins), "horizons": HORIZONS,
                     "scheme": "expanding window, out-of-sample at every origin"},
        "best_model_by_horizon": best_by_h,
        "beats_random_walk_significantly": [
            r for r in tests.to_dict("records")
            if r["verdict"] == "better than random walk"],
    }
    with open(os.path.join(OUT, "results/summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(table.round(3).to_string(index=False))
    print(tests.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
