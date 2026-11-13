"""
test_parser.py
--------------
Compares what our parser read from the PDF against the ground truth
(the correct answers saved by generate_fake_statement.py).

This is how real engineers prove their code works: not "it looks right",
but "it matched 140/140 known answers".

Run:  python test_parser.py
"""

import pandas as pd

from parse_statement import parse_pdf

truth = pd.read_csv("data/fake_statement_truth.csv", dtype={"ref_no": str})
parsed = parse_pdf("data/fake_statement.pdf")

print(f"Rows: parsed {len(parsed)} / expected {len(truth)}")
assert len(parsed) == len(truth), "Row count mismatch!"


def pct(matches):
    return f"{matches.sum()}/{len(matches)} ({100 * matches.mean():.1f}%)"


checks = {
    "Narration (wrapped lines merged)": parsed.narration == truth.narration,
    "Reference no.": parsed.ref_no == truth.ref_no,
    "Withdrawal": parsed.withdrawal.fillna(0).round(2) == truth.withdrawal.fillna(0).round(2),
    "Deposit": parsed.deposit.fillna(0).round(2) == truth.deposit.fillna(0).round(2),
    "Closing balance": parsed.closing_balance.round(2) == truth.closing_balance.round(2),
    "Counterparty (rule-based)": parsed.counterparty.str.upper() == truth.true_counterparty.str.upper(),
}
for name, m in checks.items():
    print(f"  {name:35s} {pct(m)}")

# Show any narration mistakes so we can learn from them
bad = parsed.narration != truth.narration
for i in parsed[bad].index[:5]:
    print("\n  GOT:     ", parsed.narration[i])
    print("  EXPECTED:", truth.narration[i])
