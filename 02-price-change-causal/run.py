"""
Project 2 — Measuring a rollout you could not randomize

A price change goes out to some regions and not others. Estimate its effect on
monthly cancellation rate three ways, and check whether the design holds up:

  1. two-way fixed effects difference-in-differences, clustered by region
  2. an event study, to see whether pre-trends were really parallel
  3. a synthetic control built from untreated regions, with in-space placebos

By default this runs on a simulated panel so the true effect is known. Point it
at a real panel with --data (a CSV of region, month, churn_rate, treated_from)
to run the same pipeline on live numbers.

Usage:
    python run.py
    python run.py --data data/panel.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)
OUT = os.path.dirname(os.path.abspath(__file__))
TRUE_EFFECT = 0.0040   # +0.40 percentage points on a ~10% monthly churn rate
ROLLOUT_MONTH = 25


def simulate_panel(n_regions=40, n_treated=10, months=36, rng=RNG):
    """Region-level monthly churn with region effects, seasonality, and noise."""
    regions = [f"R{i:02d}" for i in range(n_regions)]
    treated = set(regions[:n_treated])

    region_level = rng.normal(0.10, 0.02, n_regions)          # baseline churn
    season = 0.012 * np.sin(np.arange(months) * 2 * np.pi / 12)
    drift = np.linspace(0, 0.004, months)                      # common trend

    rows = []
    for i, r in enumerate(regions):
        walk = np.cumsum(rng.normal(0, 0.0003, months))        # slow region drift
        for t in range(months):
            churn = region_level[i] + season[t] + drift[t] + walk[t] + rng.normal(0, 0.0012)
            if r in treated and t >= ROLLOUT_MONTH:
                churn += TRUE_EFFECT
            rows.append({"region": r, "month": t, "churn_rate": churn,
                         "treated_from": ROLLOUT_MONTH if r in treated else np.nan})
    return pd.DataFrame(rows)


def prepare(df):
    df = df.copy()
    df["is_treated_region"] = df.treated_from.notna()
    df["post"] = df.month >= df.treated_from.fillna(np.inf)
    df["did"] = (df.is_treated_region & df.post).astype(float)
    return df


# ----------------------------------------------------------------------------
def did_twoway(df):
    """churn ~ treated_region x post + region FE + month FE, clustered SEs."""
    model = smf.ols("churn_rate ~ did + C(region) + C(month)", data=df)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": df.region})
    term = "did"
    est = fit.params[term]
    lo, hi = fit.conf_int().loc[term]
    return {"estimate": est, "ci_low": lo, "ci_high": hi,
            "p_value": float(fit.pvalues[term]), "n_obs": int(fit.nobs)}


def event_study(df, window=8):
    """Effect by month relative to rollout. Pre-period coefficients should be ~0."""
    d = df.copy()
    d["rel"] = np.where(d.is_treated_region, d.month - ROLLOUT_MONTH, np.nan)
    d["rel"] = d.rel.clip(-window, window)
    d["bin"] = np.where(d.is_treated_region, d.rel.fillna(0).astype(int), 99)
    d = d[(d.bin == 99) | (d.rel.abs() <= window)]
    d["bin"] = d.bin.replace(-1, 99)          # -1 is the reference period
    fit = smf.ols("churn_rate ~ C(bin, Treatment(99)) + C(region) + C(month)",
                  data=d).fit(cov_type="cluster", cov_kwds={"groups": d.region})
    out = []
    for k in range(-window, window + 1):
        if k == -1:
            out.append({"months_from_rollout": k, "estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0})
            continue
        term = f"C(bin, Treatment(99))[T.{k}]"
        if term in fit.params.index:
            lo, hi = fit.conf_int().loc[term]
            out.append({"months_from_rollout": k, "estimate": float(fit.params[term]),
                        "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
def synthetic_control(df, treated_units=None):
    """Weight untreated regions to match the treated group's pre-rollout path.

    Weights are non-negative and sum to one (no extrapolation), fit by NNLS on
    the pre-period series, then applied to the post-period.
    """
    wide = df.pivot_table(index="month", columns="region", values="churn_rate")
    treated_cols = treated_units or sorted(df.loc[df.is_treated_region, "region"].unique())
    donors = [c for c in wide.columns if c not in treated_cols]

    y = wide[treated_cols].mean(axis=1)
    pre = wide.index < ROLLOUT_MONTH

    # NNLS with a sum-to-one constraint imposed by an augmented row.
    penalty = 1e3
    a = np.vstack([wide.loc[pre, donors].values, penalty * np.ones(len(donors))])
    b = np.concatenate([y[pre].values, [penalty]])
    w, _ = nnls(a, b)
    if w.sum() > 0:
        w = w / w.sum()

    fitted = wide[donors].values @ w
    gap = y.values - fitted
    pre_rmse = float(np.sqrt(np.mean(gap[pre] ** 2)))
    post_effect = float(np.mean(gap[~pre]))
    weights = {d: round(float(wi), 4) for d, wi in zip(donors, w) if wi > 0.001}
    return {"post_effect": post_effect, "pre_rmse": pre_rmse, "weights": weights,
            "series": pd.DataFrame({"month": wide.index, "treated": y.values,
                                    "synthetic": fitted, "gap": gap})}


def placebo_in_space(df):
    """Run the same synthetic control pretending each untreated region was treated.

    The treated region's gap is credible only if it is large relative to these.
    """
    untreated = sorted(df.loc[~df.is_treated_region, "region"].unique())
    real = synthetic_control(df)
    effects = []
    for r in untreated:
        sub = df[(~df.is_treated_region) | (df.region == r)].copy()
        sub["is_treated_region"] = sub.region == r
        try:
            res = synthetic_control(sub, treated_units=[r])
        except Exception:
            continue
        if res["pre_rmse"] <= 5 * real["pre_rmse"]:   # drop poorly fitted placebos
            effects.append(res["post_effect"])
    effects = np.array(effects)
    bigger = int(np.sum(np.abs(effects) >= abs(real["post_effect"])))
    return {"n_placebos": int(effects.size), "placebos_at_least_as_large": bigger,
            "pseudo_p_value": round((bigger + 1) / (effects.size + 1), 4),
            "placebo_effects": [round(float(e), 5) for e in effects]}


# ----------------------------------------------------------------------------
def make_figures(sc, es):
    s = sc["series"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)

    axes[0].plot(s.month, s.treated * 100, "-", color="#0e8c6a", label="treated regions")
    axes[0].plot(s.month, s.synthetic * 100, "--", color="#5d6b73", label="synthetic control")
    axes[0].axvline(ROLLOUT_MONTH, color="#c9d1d9")
    axes[0].set_xlabel("month"); axes[0].set_ylabel("monthly churn rate (%)")
    axes[0].set_title("Treated regions vs. their synthetic control")
    axes[0].legend(frameon=False)

    axes[1].axhline(0, color="#c9d1d9")
    axes[1].errorbar(es.months_from_rollout, es.estimate * 100,
                     yerr=[(es.estimate - es.ci_low) * 100, (es.ci_high - es.estimate) * 100],
                     fmt="o", color="#0e8c6a", markersize=4, linewidth=1)
    axes[1].axvline(-0.5, color="#c9d1d9", linestyle=":")
    axes[1].set_xlabel("months from rollout"); axes[1].set_ylabel("effect on churn (pp)")
    axes[1].set_title("Event study: pre-period effects sit at zero")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures/did_and_synthetic_control.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV with region, month, churn_rate, treated_from")
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    df = pd.read_csv(args.data) if args.data else simulate_panel()
    df = prepare(df)

    did = did_twoway(df)
    es = event_study(df)
    sc = synthetic_control(df)
    placebo = placebo_in_space(df)

    es.to_csv(os.path.join(OUT, "results/event_study.csv"), index=False)
    sc["series"].to_csv(os.path.join(OUT, "results/synthetic_control_series.csv"), index=False)
    make_figures(sc, es)

    summary = {
        "data": args.data or "simulated",
        "true_effect_pp": None if args.data else round(TRUE_EFFECT * 100, 3),
        "did_estimate_pp": round(did["estimate"] * 100, 3),
        "did_ci_pp": [round(did["ci_low"] * 100, 3), round(did["ci_high"] * 100, 3)],
        "did_p_value": round(did["p_value"], 5),
        "synthetic_control_effect_pp": round(sc["post_effect"] * 100, 3),
        "synthetic_control_pre_rmse_pp": round(sc["pre_rmse"] * 100, 4),
        "donor_weights": sc["weights"],
        "placebo": {k: v for k, v in placebo.items() if k != "placebo_effects"},
        "max_pre_period_event_study_effect_pp": round(
            float(es[es.months_from_rollout < 0].estimate.abs().max() * 100), 3),
    }
    with open(os.path.join(OUT, "results/summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
