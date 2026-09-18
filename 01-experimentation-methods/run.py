"""
Project 1 — How much faster can an A/B test finish?

Compares three estimators of a treatment effect on a skewed engagement metric:

  1. difference in means            (the default)
  2. CUPED                          (adjust using pre-experiment watch time)
  3. OLS with the covariate         (regression adjustment)

and then measures how badly daily peeking inflates the false positive rate.

Everything runs on simulated data, so the true effect is known and we can check
bias, interval coverage, and power directly.

Usage:
    python run.py                 # full study (~2 min)
    python run.py --quick         # fewer replications, for a fast check
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260917)
OUT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Data generating process
# ----------------------------------------------------------------------------
def simulate_users(n, lift=0.0, rho=0.8, rng=RNG):
    """Weekly watch hours before and during an experiment.

    Watch time is lognormal (a few very heavy users dominate the mean, as in
    real streaming data). `rho` controls how strongly a user's pre-period
    behaviour predicts their in-experiment behaviour; 0.6-0.8 is typical for
    week-over-week engagement. `lift` is the true multiplicative treatment
    effect applied to the treated half.
    """
    user_level = rng.normal(0, 1, n)                     # stable user quality
    pre_noise = rng.normal(0, 1, n)
    post_noise = rng.normal(0, 1, n)

    pre_latent = rho * user_level + np.sqrt(1 - rho**2) * pre_noise
    post_latent = rho * user_level + np.sqrt(1 - rho**2) * post_noise

    pre = np.exp(1.0 + 0.8 * pre_latent)
    post = np.exp(1.0 + 0.8 * post_latent)

    treated = rng.random(n) < 0.5
    post = np.where(treated, post * (1 + lift), post)
    return pre, post, treated


# ----------------------------------------------------------------------------
# Estimators. Each returns (estimate, standard error) of the absolute effect.
# ----------------------------------------------------------------------------
def diff_in_means(pre, post, treated):
    a, b = post[treated], post[~treated]
    est = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return est, se


def cuped(pre, post, treated):
    """Subtract the part of the outcome explained by pre-period behaviour.

    theta is the OLS slope of post on pre (pooled across arms, which keeps the
    estimator unbiased because pre-period data cannot be affected by treatment).
    """
    theta = np.cov(post, pre, ddof=1)[0, 1] / pre.var(ddof=1)
    adjusted = post - theta * (pre - pre.mean())
    return diff_in_means(pre, adjusted, treated)


def ols_adjusted(pre, post, treated):
    """Regress outcome on treatment and the centred covariate."""
    x = np.column_stack([np.ones(pre.size), treated.astype(float), pre - pre.mean()])
    beta, *_ = np.linalg.lstsq(x, post, rcond=None)
    resid = post - x @ beta
    dof = pre.size - x.shape[1]
    xtx_inv = np.linalg.inv(x.T @ x)
    sigma2 = resid @ resid / dof
    se = np.sqrt(sigma2 * xtx_inv[1, 1])
    return beta[1], se


ESTIMATORS = {
    "difference in means": diff_in_means,
    "CUPED": cuped,
    "OLS adjusted": ols_adjusted,
}


# ----------------------------------------------------------------------------
# Study 1: bias, coverage, power
# ----------------------------------------------------------------------------
def run_study(sample_sizes, lift, reps, rho=0.8):
    rows = []
    baseline_mean = np.exp(1.0 + 0.5 * 0.8**2)  # E[exp(mu + sigma*Z)]
    true_effect = baseline_mean * lift

    for n in sample_sizes:
        acc = {name: {"est": [], "se": [], "cover": [], "reject": []} for name in ESTIMATORS}
        for _ in range(reps):
            pre, post, treated = simulate_users(n, lift=lift, rho=rho)
            for name, fn in ESTIMATORS.items():
                est, se = fn(pre, post, treated)
                lo, hi = est - 1.96 * se, est + 1.96 * se
                acc[name]["est"].append(est)
                acc[name]["se"].append(se)
                acc[name]["cover"].append(lo <= true_effect <= hi)
                acc[name]["reject"].append(lo > 0 or hi < 0)
        for name, a in acc.items():
            rows.append({
                "n": n,
                "estimator": name,
                "mean_estimate": np.mean(a["est"]),
                "true_effect": true_effect,
                "bias": np.mean(a["est"]) - true_effect,
                "sd_of_estimate": np.std(a["est"], ddof=1),
                "mean_se": np.mean(a["se"]),
                "coverage_95": np.mean(a["cover"]),
                "power": np.mean(a["reject"]),
            })
    return pd.DataFrame(rows)


def required_n(df, estimator, target=0.80):
    """Smallest sample size reaching `target` power, by linear interpolation."""
    sub = df[df.estimator == estimator].sort_values("n")
    n, p = sub.n.values, sub.power.values
    for i in range(1, len(n)):
        if p[i] >= target > p[i - 1]:
            w = (target - p[i - 1]) / (p[i] - p[i - 1])
            return n[i - 1] + w * (n[i] - n[i - 1])
    return np.nan


# ----------------------------------------------------------------------------
# Study 2: what daily peeking costs you
# ----------------------------------------------------------------------------
def peeking_study(n_per_day=2000, days=14, reps=2000):
    """No true effect. How often does a daily-monitored test hit p < 0.05?"""
    fixed_hits = 0
    peek_hits = 0
    for _ in range(reps):
        pre, post, treated = simulate_users(n_per_day * days, lift=0.0)
        stopped = False
        for d in range(1, days + 1):
            k = n_per_day * d
            est, se = cuped(pre[:k], post[:k], treated[:k])
            if abs(est / se) > 1.96:
                stopped = True
        est, se = cuped(pre, post, treated)
        fixed_hits += abs(est / se) > 1.96
        peek_hits += stopped
    return {
        "replications": reps,
        "false_positive_rate_fixed_horizon": fixed_hits / reps,
        "false_positive_rate_daily_peeking": peek_hits / reps,
        "days_monitored": days,
    }


# ----------------------------------------------------------------------------
def make_figure(df, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=160)
    styles = {"difference in means": "o-", "CUPED": "s-", "OLS adjusted": "^--"}
    for name, style in styles.items():
        sub = df[df.estimator == name].sort_values("n")
        ax.plot(sub.n, sub.power, style, label=name, linewidth=1.8, markersize=5)
    ax.axhline(0.8, color="#8a96a3", linestyle=":", linewidth=1.2)
    ax.annotate("80% power", (df.n.min(), 0.815), color="#5d6b73", fontsize=9)
    ax.set_xlabel("users per experiment")
    ax.set_ylabel("power to detect the true lift")
    ax.set_title("Pre-experiment data reaches the same answer with fewer users")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    reps = 100 if args.quick else 400
    peek_reps = 200 if args.quick else 1500
    sizes = [8000, 16000, 32000, 64000, 96000, 128000]
    lift = 0.02  # a 2% lift in mean watch time

    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    df = run_study(sizes, lift=lift, reps=reps)
    df.to_csv(os.path.join(OUT, "results/power_and_coverage.csv"), index=False)
    make_figure(df, os.path.join(OUT, "figures/power_curves.png"))

    base = required_n(df, "difference in means")
    summary = {
        "true_lift": lift,
        "replications_per_cell": reps,
        "required_n_for_80pct_power": {
            name: (None if np.isnan(required_n(df, name)) else round(required_n(df, name)))
            for name in ESTIMATORS
        },
        "sample_size_reduction_vs_difference_in_means": {
            name: (None if np.isnan(required_n(df, name)) or np.isnan(base)
                   else round(100 * (required_n(df, name) - base) / base, 1))
            for name in ESTIMATORS
        },
        "peeking": peeking_study(reps=peek_reps),
    }
    with open(os.path.join(OUT, "results/summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.set_option("display.width", 120)
    print(df.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
