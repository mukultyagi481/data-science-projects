# Deciding what is worth A/B testing

**Question.** Online tests are expensive and slow, offline metrics are cheap and misleading. Given three candidate rankers, which one earns a slot in the test queue, and what test would settle it?

**Setup.** Implicit feedback (204,262 training interactions, 50,116 held out) split by a global time cutoff, so no model sees the future. 3,140 users evaluated. Metrics: NDCG@10, Recall@10, and catalogue coverage, each with a bootstrap interval over users, plus a breakdown by how much history each user has. Runs on simulated interactions by default; pass `--data data/ratings.csv` to run the same pipeline on MovieLens.

| Model | NDCG@10 | 95% interval | Recall@10 | Catalogue coverage |
|---|---|---|---|---|
| Popularity | 0.178 | [0.172, 0.185] | 0.151 | 1.4% |
| Item-to-item cosine | 0.201 | [0.194, 0.208] | 0.172 | 20.4% |
| ALS, 32 factors | 0.129 | [0.124, 0.134] | 0.119 | 58.1% |

**What I found.** Item-to-item cosine beat the popularity baseline by 12.8% on NDCG@10, with non-overlapping intervals, and recommended from 15 times as much of the catalogue. Untuned ALS lost to popularity on ranking quality while covering the most catalogue — a useful reminder that the more sophisticated model is not automatically the better one, and that factor count, regularisation, and confidence weighting need tuning before a fair comparison. Gains also concentrate: the lift over popularity is largest for users with short histories and shrinks for the heaviest users, whose future views are already close to the popular set.

**The online test this implies.**
- Hypothesis: replacing the popularity row with item-to-item recommendations increases member engagement without reducing catalogue breadth.
- Primary metric: days active in the first 14 days after exposure (chosen in project 3 for sensitivity and retention prediction).
- Secondary: titles started per member, share of plays from outside the top 100 titles.
- Guardrails: 30-day retention, playback error rate, latency of the recommendations call.
- Design: 50/50 member-level assignment, two weeks minimum to cover a full weekly cycle, CUPED-adjusted on pre-period activity, sample sized from project 1's power curves.
- Decision rule: ship if the primary metric gains at least the pre-agreed minimum detectable effect with no guardrail regression; otherwise iterate offline.

**Limits.** Offline ranking metrics reward predicting what a member would have watched anyway, not what a different recommendation would have caused them to watch. They cannot see novelty effects or how the row competes with the rest of the page. That gap is the whole reason the online test exists.

## Run it
```bash
pip install -r requirements.txt
python run.py
python run.py --data data/ratings.csv --max-users 20000
```
MovieLens 25M: https://grouplens.org/datasets/movielens/25m/ — unzip and put `ratings.csv` in `data/`.

Outputs: `results/summary.json`, `results/offline_metrics.csv`, `results/by_history.csv`, `figures/offline_evaluation.png`.
