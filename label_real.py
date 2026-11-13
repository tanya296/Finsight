"""
label_real.py -- makes a file for YOU to hand-label real transactions.

It reads the real statement PDF and writes real/real_labels.csv with an EMPTY
'my_category' column. You fill in the correct category for each row.
These hand-labels are the "ground truth" for the real-data test.

Everything stays in the real/ folder (never uploaded, never on GitHub).

Run:  python label_real.py "real/<your statement>.pdf"
"""

import sys
from pathlib import Path

import pandas as pd

from categories import CATEGORIES
from parse_statement import parse_pdf

OUT = Path("real/real_labels.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python label_real.py "real/<statement>.pdf"')
    df = parse_pdf(sys.argv[1])[["date", "narration", "amount"]]
    df["my_category"] = ""

    if OUT.exists():                       # keep labels you already typed
        old = pd.read_csv(OUT, dtype=str).fillna("")
        done = dict(zip(old.narration, old.my_category))
        df["my_category"] = [done.get(n, "") for n in df.narration]
        df = pd.concat([old, df[~df.narration.isin(old.narration)]], ignore_index=True)

    df.to_csv(OUT, index=False)
    todo = (df.my_category == "").sum()
    print(f"Wrote {OUT}: {len(df)} transactions, {todo} still need a label.\n")
    print("Open it in Excel or VS Code and type ONE of these in the my_category column:")
    for c, d in CATEGORIES.items():
        print(f"  {c:14s} {d}")
    print("\nSpelling must match exactly (e.g. 'Food', not 'food'). Save as CSV.")
