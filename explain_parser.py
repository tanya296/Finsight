"""
explain_parser.py
-----------------
Builds a tiny 2-transaction fake statement (same layout as HDFC) and
runs the parser STEP BY STEP, printing what it sees at each stage.

Run:  python explain_parser.py
"""

import sqlite3

import pandas as pd
import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import parse_statement as P

PDF = "data/mini_example.pdf"

# ---------------------------------------------------------------
# Make the mini PDF: 2 transactions, narrations wrapped like the real one
# ---------------------------------------------------------------
COLS = [("Date", 30, 75), ("Narration", 75, 255), ("Chq./Ref.No.", 255, 345),
        ("Value Dt", 345, 390), ("Withdrawal Amt.", 390, 455),
        ("Deposit Amt.", 455, 515), ("Closing Balance", 515, 575)]
c = canvas.Canvas(PDF, pagesize=A4)
c.setFont("Helvetica-Bold", 7)
c.rect(30, 700, 545, 18)
for name, x0, x1 in COLS:
    c.drawCentredString((x0 + x1) / 2, 706, name)
c.setFont("Helvetica", 7)
rows = [  # (y, date, narration lines, ref, withdrawal, deposit, balance)
    (688, "01/09/26", ["UPI-S PRIYA-PRIYA12345@OKICICI-SBIN00", "04567-551234098712-UPI"],
     "0000551234098712", "", "50,000.00", "62,345.67"),
    (662, "01/09/26", ["FT-XX1234-IMPS TRANSACTION-RAVI", "CHANDR", "AN-G51239876543"],
     "0000G51239876543", "50,000.00", "", "12,345.67"),
]
for y, d, narr, ref, wd, dep, bal in rows:
    c.drawString(33, y, d)
    for i, line in enumerate(narr):
        c.drawString(78, y - i * 11, line)
    c.drawRightString(342, y, ref)
    c.drawString(349, y, d)
    c.drawRightString(452, y, wd)
    c.drawRightString(512, y, dep)
    c.drawRightString(572, y, bal)
for _, x0, _ in COLS:
    c.line(x0, 718, x0, 625)
c.line(575, 718, 575, 625)
c.save()


def title(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


pdf = pdfplumber.open(PDF)
page = pdf.pages[0]

# ---------------------------------------------------------------
title("STEP 0: What a PDF really contains -> words + coordinates")
print("A PDF has NO rows or columns. Just words placed at (x, top) positions.\n")
print(f"{'word':40s} {'x0':>6s} {'x1':>6s} {'top':>6s}")
for w in page.extract_words(x_tolerance=1.5)[:22]:
    print(f"{w['text']:40s} {w['x0']:6.0f} {w['x1']:6.0f} {w['top']:6.0f}")
print("... (x = how far from the LEFT, top = how far from the TOP)")

# ---------------------------------------------------------------
title("STEP 1: find_columns() -> where is each column?")
cols, header_bottom = P.find_columns(page)
xs = sorted({round(e["x0"]) for e in page.edges if e.get("orientation") == "v"})
print(f"Found the word 'Narration' -> header row is around top={header_bottom:.0f}")
print(f"Vertical table lines found at x = {xs}")
print("So the columns are:")
for name, l, r in cols:
    print(f"   {name:16s} from x={l:<4.0f} to x={r:.0f}")

# ---------------------------------------------------------------
title("STEP 2: page_lines() -> group words into LINES, put them in COLUMNS")
print("Words with (almost) the same 'top' = same line.")
print("Each word goes into the column whose x-range contains it.\n")
lines = P.page_lines(page, cols, header_bottom)
for i, cells in enumerate(lines, 1):
    print(f"Line {i}: {cells}")

# ---------------------------------------------------------------
title("STEP 3: merge_lines() -> the DATE-ANCHOR rule")
for i, cells in enumerate(lines, 1):
    d = cells.get("date", "")
    if P.DATE_RE.match(d):
        print(f"Line {i}: has date '{d}'  -> START a new transaction")
    else:
        print(f"Line {i}: no date        -> ADD '{cells.get('narration')}' to previous narration")
txns = P.merge_lines(lines)
print("\nResult: 5 lines became", len(txns), "transactions:")
for t in txns:
    print("  narration =", t["narration"])

# ---------------------------------------------------------------
title("STEP 4: classify() -> transaction type + counterparty")
for t in txns:
    typ, who = P.classify(t["narration"])
    print(f"  {t['narration'][:45]:47s} -> type={typ}, counterparty='{who}'")

# ---------------------------------------------------------------
title("STEP 5: clean numbers + dates, then the BALANCE CHECK")
df = P.parse_pdf(PDF)
print(df[["date", "withdrawal", "deposit", "amount", "closing_balance"]].to_string(index=False))
print(f"\n  Check: {df.closing_balance[0]:,.2f} + ({df.amount[1]:,.2f}) "
      f"= {df.closing_balance[0] + df.amount[1]:,.2f}  "
      f"vs statement says {df.closing_balance[1]:,.2f}  -> "
      f"{'OK' if abs(df.closing_balance[0] + df.amount[1] - df.closing_balance[1]) < 0.01 else 'MISMATCH'}")

# ---------------------------------------------------------------
title("STEP 6: save_sqlite() -> stored as rows in a database table")
with sqlite3.connect("data/mini_example.db") as con:
    df.to_sql("transactions", con, if_exists="replace", index=False)
    out = pd.read_sql("SELECT date, txn_type, counterparty, amount, closing_balance "
                      "FROM transactions", con)
print("SELECT date, txn_type, counterparty, amount, closing_balance FROM transactions;\n")
print(out.to_string(index=False))
