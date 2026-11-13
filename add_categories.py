"""
add_categories.py -- adds 'category' and 'category_source' columns to
data/transactions.db using the HYBRID categoriser (rules + your model).
The Day 4 agent needs this to answer questions like
"How much did we spend on Food?".

Run:  python add_categories.py qwen-1.5b
"""

import sqlite3
import sys

import pandas as pd

from hybrid import Hybrid

DB = "data/transactions.db"

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen-1.5b"
    with sqlite3.connect(DB) as con:
        df = pd.read_sql("SELECT * FROM transactions", con)
        df["category"], df["category_source"] = Hybrid(name).predict(list(df.narration))
        df.to_sql("transactions", con, if_exists="replace", index=False)
    print(f"\nAdded 'category' to {len(df)} rows in {DB} "
          f"(sources: {df.category_source.value_counts().to_dict()}). Spending by category:")
    spend = df[df.amount < 0].groupby("category").amount.sum().abs().sort_values(ascending=False)
    print(spend.round(2).to_string())
