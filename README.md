# Data science projects — experimentation and causal inference

Four self-contained projects, each answering a product question rather than demonstrating a technique. Every project runs end to end with `python run.py`, writes its numbers to `results/`, and opens with a short memo.

| # | Project | Question | Methods |
|---|---|---|---|
| 1 | [Experimentation methods](01-experimentation-methods) | How much faster can an A/B test finish? | CUPED, regression adjustment, power simulation, peeking |
| 2 | [Price change, causal](02-price-change-causal) | What did a rollout we could not randomize actually do? | Difference-in-differences, event study, synthetic control, placebos |
| 3 | [Metric design and retention](03-metric-design-retention) | Which early metric deserves to be the primary one? | Sensitivity analysis, Kaplan–Meier, Cox model |
| 4 | [Recommender offline evaluation](04-recommender-offline-eval) | Which model is worth testing online, and how? | Time-based split, NDCG/Recall, bootstrap, ALS, test design |
| 5 | [Demand forecasting with SARIMAX](05-demand-forecasting-sarimax) | Does forecasting economic drivers improve a demand forecast? | SARIMAX, exogenous regressors, rolling-origin backtesting, model selection |

## A note on the data

Projects 1, 2, 3, and 5 run on simulated data. That is a deliberate choice, not a shortcut: simulation is the only way to know the true effect and therefore the only way to check whether an estimator is biased or its intervals are honest. Each README says plainly which numbers come from simulation. Projects 2 and 4 also accept real data (a regional panel CSV, or MovieLens) through a command-line flag, and running them on real data is the natural next step.

## How to work through these

Each project is independent. Read the memo first — that is the work sample — then the code.
