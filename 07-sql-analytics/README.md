# SQL analytics on a transactional schema

**What this is.** Six analytical queries against a normalised transactional database (customers, invoices, invoice lines, tracks, genres), ordered the way an analysis actually proceeds. Each query opens with the business question it answers and the technique it uses. The queries in `queries/` are the work product; `run.py` just executes them and writes each result to `results/`.

| Query | Question | Technique |
|---|---|---|
| [01_revenue_trend](queries/01_revenue_trend.sql) | How is revenue trending, and is the latest month unusual? | Monthly rollup, `LAG` for MoM and YoY, trailing three-month window average |
| [02_cohort_retention](queries/02_cohort_retention.sql) | Do customers acquired in different quarters keep buying at different rates? | First-purchase cohorts, quarters-since-signup buckets, share of original cohort |
| [03_top_genres_by_country](queries/03_top_genres_by_country.sql) | Which genres should we promote in each major market? | `RANK()` within country partitions, with a minimum-revenue floor |
| [04_value_concentration](queries/04_value_concentration.sql) | Is revenue concentrated in a few customers or spread across the base? | `NTILE(10)` value deciles, cumulative window sum |
| [05_purchase_gaps](queries/05_purchase_gaps.sql) | How long is a *normal* gap between orders, and who is overdue? | `LAG` over each customer's history, per-customer baseline, lapse flags |
| [06_data_quality_checks](queries/06_data_quality_checks.sql) | What in the data could make all of the above wrong? | Reconciliation of invoice totals against line items, orphan-key and null checks |

**Two findings worth calling out.**

*Taste is local, and a global top-genre list would mislead.* Rock leads every large market, but the second and third slots differ sharply: Latin is 27.6% of Brazilian revenue, Alternative & Punk is 15.7% in France, and the Czech Republic's second and third genres are TV Shows and Drama rather than music at all. A single global promotion list would be wrong in most markets.

*Revenue is almost perfectly uniform across value deciles* — the top decile holds 12.0% of revenue where a real consumer business typically sees 40% or more. That is a property of this sample database, not a business insight, and it is exactly the kind of thing worth catching before anyone builds a retention strategy on it. Which brings up the honest caveat below.

**About the data.** This is the Chinook sample database, a standard public schema used for SQL practice. It is realistic in *shape* — proper foreign keys, line-item granularity, five years of invoices — but the values are generated, so the business conclusions above are not real-world findings. The queries are the transferable part: the same cohort, ranking, concentration, and reconciliation patterns run unchanged against a real warehouse. Project 6 in this repository is the one built on real data.

**Why the last query matters most.** Every result above assumes invoice totals reconcile with their line items and that keys join cleanly. `06_data_quality_checks` tests those assumptions and returns one row per check, so it reads as a pass/fail report. On this database all six checks pass. On a real warehouse they usually do not, and finding that out after presenting the numbers is the expensive way to learn it.

## Run it
```bash
pip install -r requirements.txt
python run.py                 # runs all six, downloads the database on first run
python run.py --query 03      # run just one
```
Outputs: one CSV per query in `results/`.
