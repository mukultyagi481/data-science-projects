"""
Project 3 — Choosing a metric you can still defend in six months

Two questions a team actually argues about:

  1. Which early engagement metric is sensitive enough to move an experiment?
     (a metric with huge variance needs impractical sample sizes)
  2. Which one actually predicts whether the member is still here in 90 days?

A metric only earns the "primary metric" label if it scores on both. This script
evaluates five candidates measured in a member's first 14 days, then models
time-to-cancel with Kaplan-Meier curves and a Cox model.

Usage:
    python run.py
"""

import json
import os

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(11)
OUT = os.path.dirname(os.path.abspath(__file__))
N_USERS = 40_000
HORIZON = 90


def simulate_cohort(n=N_USERS, rng=RNG):
    """Sign-up cohort with 14 days of activity and a time-to-cancel outcome."""
    engagement = rng.normal(0, 1, n)                       # latent taste fit
    device = rng.choice(["TV", "mobile", "web"], n, p=[0.45, 0.4, 0.15])
    channel = rng.choice(["organic", "paid", "partner"], n, p=[0.5, 0.35, 0.15])

    device_effect = np.select([device == "TV", device == "mobile"], [0.35, -0.1], 0.0)
    channel_effect = np.select([channel == "paid", channel == "partner"], [-0.25, 0.1], 0.0)
    latent = engagement + device_effect + channel_effect

    # First 14 days of behaviour
    days_active = rng.binomial(14, 1 / (1 + np.exp(-(0.2 + 0.7 * latent))))
    sessions = rng.poisson(np.exp(0.9 + 0.45 * latent))
    hours = rng.lognormal(0.4 + 0.55 * latent, 0.8)        # heavy right tail
    titles = rng.poisson(np.exp(0.5 + 0.4 * latent))
    watched_3_titles = (titles >= 3).astype(int)

    # Time to cancel: higher latent engagement, lower hazard
    hazard = 0.006 * np.exp(-0.9 * latent)
    t = rng.exponential(1 / hazard)
    observed_days = np.minimum(t, HORIZON)
    churned = (t <= HORIZON).astype(int)

    return pd.DataFrame({
        "device": device, "channel": channel,
        "days_active_14d": days_active, "sessions_14d": sessions,
        "hours_14d": hours, "titles_14d": titles,
        "watched_3_titles_14d": watched_3_titles,
        "days_observed": observed_days, "churned_90d": churned,
        "retained_90d": 1 - churned,
    })


CANDIDATES = ["hours_14d", "days_active_14d", "sessions_14d", "titles_14d",
              "watched_3_titles_14d"]


def sensitivity(df, n_per_arm=50_000):
    """Minimum detectable effect at 80% power, 5% two-sided, for each metric.

    MDE in relative terms is 2.8 * CV * sqrt(2/n), where CV is the coefficient
    of variation. Noisy, heavy-tailed metrics have a large CV and therefore need
    much bigger effects (or samples) before a test can see anything.
    """
    rows = []
    for m in CANDIDATES:
        x = df[m].astype(float)
        cv = x.std(ddof=1) / x.mean()
        mde_rel = 2.802 * cv * np.sqrt(2 / n_per_arm)
        rows.append({"metric": m, "mean": x.mean(), "sd": x.std(ddof=1),
                     "coef_of_variation": cv,
                     "mde_relative_pct_at_50k_per_arm": 100 * mde_rel})
    return pd.DataFrame(rows).sort_values("mde_relative_pct_at_50k_per_arm")


def predictive_value(df):
    """Does the early metric separate members who stay from members who leave?"""
    rows = []
    for m in CANDIDATES:
        auc = roc_auc_score(df.retained_90d, df[m])
        q = pd.qcut(df[m].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        by_q = df.groupby(q, observed=True).retained_90d.mean()
        rows.append({"metric": m, "auc_retained_90d": auc,
                     "retention_bottom_quintile": by_q.iloc[0],
                     "retention_top_quintile": by_q.iloc[-1],
                     "top_minus_bottom_pp": 100 * (by_q.iloc[-1] - by_q.iloc[0])})
    return pd.DataFrame(rows).sort_values("auc_retained_90d", ascending=False)


def survival_models(df):
    """Kaplan-Meier by device, then a Cox model with the early metrics."""
    km = {}
    kmf = KaplanMeierFitter()
    curves = {}
    for d, sub in df.groupby("device"):
        kmf.fit(sub.days_observed, sub.churned_90d, label=d)
        km[d] = float(kmf.predict(HORIZON - 1))
        curves[d] = kmf.survival_function_.iloc[:, 0]

    cox_df = pd.get_dummies(
        df[["days_observed", "churned_90d", "days_active_14d", "hours_14d",
            "device", "channel"]],
        columns=["device", "channel"], drop_first=True, dtype=float)
    cox_df["hours_14d"] = np.log1p(cox_df.hours_14d)       # tame the tail
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="days_observed", event_col="churned_90d")
    hr = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                      "exp(coef) upper 95%", "p"]].round(4)
    c_index = concordance_index(cox_df.days_observed, -cph.predict_partial_hazard(cox_df),
                               cox_df.churned_90d)
    return km, curves, hr, float(c_index)


def make_figure(sens, pred, curves):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=160)

    s = sens.sort_values("mde_relative_pct_at_50k_per_arm")
    axes[0].barh(s.metric, s.mde_relative_pct_at_50k_per_arm, color="#0e8c6a")
    axes[0].set_xlabel("smallest detectable lift (%)")
    axes[0].set_title("Sensitivity at 50k per arm")
    axes[0].invert_yaxis()

    p = pred.sort_values("auc_retained_90d")
    axes[1].barh(p.metric, p.auc_retained_90d, color="#101820")
    axes[1].set_xlim(0.5, max(0.75, p.auc_retained_90d.max() + 0.03))
    axes[1].set_xlabel("AUC for 90-day retention")
    axes[1].set_title("Predictive value")

    for label, curve in curves.items():
        axes[2].step(curve.index, curve.values, where="post", label=label)
    axes[2].set_xlabel("days since sign-up")
    axes[2].set_ylabel("still subscribed")
    axes[2].set_title("Survival by primary device")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures/metric_design.png"))
    plt.close(fig)


def main():
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    df = simulate_cohort()
    sens = sensitivity(df)
    pred = predictive_value(df)
    km, curves, hr, c_index = survival_models(df)

    sens.to_csv(os.path.join(OUT, "results/sensitivity.csv"), index=False)
    pred.to_csv(os.path.join(OUT, "results/predictive_value.csv"), index=False)
    hr.to_csv(os.path.join(OUT, "results/cox_hazard_ratios.csv"))
    make_figure(sens, pred, curves)

    scored = sens.merge(pred, on="metric")
    scored["rank_sensitivity"] = scored.mde_relative_pct_at_50k_per_arm.rank()
    scored["rank_prediction"] = scored.auc_retained_90d.rank(ascending=False)
    scored["combined"] = scored.rank_sensitivity + scored.rank_prediction
    recommendation = scored.sort_values("combined").iloc[0]

    summary = {
        "n_users": int(len(df)),
        "overall_90d_churn": round(float(df.churned_90d.mean()), 4),
        "recommended_primary_metric": recommendation.metric,
        "why": {
            "smallest_detectable_lift_pct": round(float(recommendation.mde_relative_pct_at_50k_per_arm), 3),
            "auc_for_90d_retention": round(float(recommendation.auc_retained_90d), 4),
        },
        "survival_at_90_days_by_device": {k: round(v, 4) for k, v in km.items()},
        "cox_concordance": round(c_index, 4),
        "metric_table": scored[["metric", "mde_relative_pct_at_50k_per_arm",
                                "auc_retained_90d", "top_minus_bottom_pp"]]
                          .round(4).to_dict("records"),
    }
    with open(os.path.join(OUT, "results/summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(sens.round(4).to_string(index=False))
    print(pred.round(4).to_string(index=False))
    print(hr.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
