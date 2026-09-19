"""
Project 7 — SQL analytics on a real transactional schema

Downloads the Chinook sample database (a public music-store schema: customers,
invoices, invoice lines, tracks, genres), runs every query in queries/, and
writes each result to results/ as CSV.

The queries are the work product. Each one opens with the business question it
answers and the technique it uses, and they are ordered the way an analysis
actually proceeds: trend, cohorts, segmentation, concentration, lapse risk, and
finally the data-quality checks that decide whether any of it can be reported.

Usage:
    python run.py                 # run every query
    python run.py --query 02      # run one
"""

import argparse
import glob
import os
import sqlite3

import pandas as pd
import requests

OUT = os.path.dirname(os.path.abspath(__file__))
DB_URL = ("https://raw.githubusercontent.com/lerocha/chinook-database/master/"
          "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite")
DB_PATH = os.path.join(OUT, "data", "chinook.sqlite")


def ensure_db():
    if os.path.exists(DB_PATH):
        return DB_PATH
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print("downloading Chinook database ...")
    r = requests.get(DB_URL, timeout=120)
    r.raise_for_status()
    with open(DB_PATH, "wb") as f:
        f.write(r.content)
    return DB_PATH


def run_query(conn, path):
    with open(path) as f:
        sql = f.read()
    df = pd.read_sql_query(sql, conn)
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    df.to_csv(os.path.join(OUT, "results", f"{name}.csv"), index=False)
    return name, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="run only queries whose filename starts with this")
    args = ap.parse_args()

    conn = sqlite3.connect(ensure_db())
    paths = sorted(glob.glob(os.path.join(OUT, "queries", "*.sql")))
    if args.query:
        paths = [p for p in paths if os.path.basename(p).startswith(args.query)]

    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 30)

    for path in paths:
        name, df = run_query(conn, path)
        print(f"\n{'=' * 78}\n{name}  ({len(df)} rows)\n{'=' * 78}")
        print(df.head(12).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
