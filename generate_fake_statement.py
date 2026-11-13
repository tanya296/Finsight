"""
generate_fake_statement.py
--------------------------
Creates a FAKE bank statement PDF that looks like an HDFC statement,
plus a "ground truth" CSV with the exact correct data.

Why fake data?
  - Real statements are private. We never put them on GitHub.
  - Because WE generated it, we know the correct answer for every row,
    so we can test whether our parser is 100% right.

Run:  python generate_fake_statement.py
Output: data/fake_statement.pdf  and  data/fake_statement_truth.csv
"""

import random
import textwrap
from datetime import date, timedelta

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

random.seed(42)  # same "random" data every time -> reproducible results

# ---------------------------------------------------------------
# 1. Fake merchants, people and transaction templates
# ---------------------------------------------------------------
# (merchant name, UPI handle, category, min amount, max amount)
MERCHANTS = [
    ("SWIGGY", "swiggy8@ybl", "Food", 150, 900),
    ("ZOMATO", "zomato.order@hdfcbank", "Food", 150, 1000),
    ("BLINKIT", "blinkit.payu@axisbank", "Groceries", 100, 1500),
    ("BIGBASKET", "bigbasket@icici", "Groceries", 300, 3000),
    ("AMAZON PAY", "amazonpay@apl", "Shopping", 200, 5000),
    ("FLIPKART", "flipkart.pay@axisbank", "Shopping", 300, 6000),
    ("UBER INDIA", "uber.rides@hdfcbank", "Travel", 80, 700),
    ("OLA CABS", "olacabs@ybl", "Travel", 80, 600),
    ("IRCTC", "irctc.uts@sbi", "Travel", 200, 3000),
    ("NETFLIX", "netflix.upi@icici", "Subscriptions", 199, 649),
    ("SPOTIFY", "spotify.in@hdfcbank", "Subscriptions", 119, 179),
    ("JIO RECHARGE", "jio.prepaid@sbi", "Bills", 239, 999),
    ("BESCOM", "bescom.bill@okaxis", "Bills", 600, 2500),
    ("APOLLO PHARMACY", "apollopharmacy@ybl", "Health", 100, 2000),
    ("BOOKMYSHOW", "bookmyshow@paytm", "Entertainment", 200, 1200),
    ("DMART", "dmart.ready@okicici", "Groceries", 500, 4000),
]

# Fake people (for person-to-person UPI / IMPS transfers)
PEOPLE = ["RAHUL SHARMA", "PRIYA NAIR", "ANIL KUMAR", "SUNITA DEVI",
          "VIKRAM SINGH", "MEERA IYER", "ARJUN REDDY", "KAVYA MENON"]

BANKS = ["SBIN", "ICIC", "UTIB", "HDFC", "KKBK", "IOBA"]


def ref_no():
    """A random 12-15 digit reference number like real statements have."""
    return str(random.randint(10**11, 10**15 - 1)).zfill(16)


def make_narration(kind):
    """
    Build a narration string in the style HDFC uses.
    Returns (narration, is_debit, amount, true_category, true_counterparty)
    """
    r = ref_no()[-12:]
    if kind == "upi_merchant":
        name, handle, cat, lo, hi = random.choice(MERCHANTS)
        amt = round(random.uniform(lo, hi), 2)
        narr = f"UPI-{name}-{handle.upper()}-{random.choice(BANKS)}0{random.randint(1000,9999)}-{r}-UPI"
        return narr, True, amt, cat, name
    if kind == "upi_person_out":
        p = random.choice(PEOPLE)
        handle = p.split()[0] + str(random.randint(100, 99999)) + "@OKSBI"
        amt = round(random.choice([100, 200, 500, 1000, 1500, 2000]) + random.random() * 0, 2)
        narr = f"UPI-{p}-{handle}-{random.choice(BANKS)}0{random.randint(1000,9999)}-{r}-PAYMENT"
        return narr, True, amt, "Transfers", p
    if kind == "upi_person_in":
        p = random.choice(PEOPLE)
        handle = p.split()[0] + str(random.randint(100, 99999)) + "@OKICICI"
        amt = float(random.choice([500, 1000, 2000, 5000]))
        narr = f"UPI-S {p}-{handle}-{random.choice(BANKS)}0{random.randint(1000,9999)}-{r}-UPI"
        return narr, False, amt, "Transfers", p
    if kind == "imps_out":
        p = random.choice(PEOPLE)
        amt = float(random.choice([5000, 10000, 20000]))
        narr = f"FT-XX{random.randint(1000,9999)}-IMPS TRANSACTION-{p}"
        return narr, True, amt, "Transfers", p
    if kind == "salary":
        amt = 85000.00
        narr = f"NEFT CR-CITI0000002-ACME TECHNOLOGIES PVT LTD-SALARY FOR THE MONTH-{r}"
        return narr, False, amt, "Income", "ACME TECHNOLOGIES PVT LTD"
    if kind == "atm":
        amt = float(random.choice([1000, 2000, 3000, 5000]))
        narr = f"ATW-512967XXXXXX{random.randint(1000,9999)}-S1AW{random.randint(1000,9999)}-BANGALORE"
        return narr, True, amt, "Cash", "ATM"
    if kind == "card_pos":
        name, _, cat, lo, hi = random.choice(MERCHANTS)
        amt = round(random.uniform(lo, hi), 2)
        narr = f"POS 512967XXXXXX{random.randint(1000,9999)} {name} BANGALORE"
        return narr, True, amt, cat, name
    if kind == "interest":
        amt = round(random.uniform(50, 400), 2)
        narr = "CREDIT INTEREST CAPITALISED"
        return narr, False, amt, "Income", "BANK"
    raise ValueError(kind)


# ---------------------------------------------------------------
# 2. Create ~2 months of transactions
# ---------------------------------------------------------------
def generate_transactions(start=date(2026, 9, 1), days=61, opening=25000.00):
    rows = []
    balance = opening
    weights = {"upi_merchant": 60, "upi_person_out": 8, "upi_person_in": 6,
               "imps_out": 2, "atm": 3, "card_pos": 8}

    for d in range(days):
        day = start + timedelta(days=d)
        kinds = []
        if day.day == 1:
            kinds.append("salary")
        if day.day == 30:
            kinds.append("interest")
        # 0-4 random transactions per day
        kinds += random.choices(list(weights), weights=list(weights.values()),
                                k=random.randint(0, 4))
        for kind in kinds:
            narr, is_debit, amt, cat, who = make_narration(kind)
            if is_debit and amt > balance:  # don't go negative
                continue
            balance = round(balance - amt if is_debit else balance + amt, 2)
            rows.append({
                "date": day.strftime("%d/%m/%y"),
                "narration": narr,
                "ref_no": ref_no(),
                "value_date": day.strftime("%d/%m/%y"),
                "withdrawal": amt if is_debit else None,
                "deposit": None if is_debit else amt,
                "closing_balance": balance,
                "true_category": cat,
                "true_counterparty": who,
            })
    return rows


# ---------------------------------------------------------------
# 3. Draw the PDF (mimicking HDFC's table layout)
# ---------------------------------------------------------------
# x positions of the column borders (in PDF points; A4 width = 595)
COLS = [
    ("Date", 30, 75),
    ("Narration", 75, 255),
    ("Chq./Ref.No.", 255, 345),
    ("Value Dt", 345, 390),
    ("Withdrawal Amt.", 390, 455),
    ("Deposit Amt.", 455, 515),
    ("Closing Balance", 515, 575),
]
LINE_H = 11          # height of one text line
NARR_WIDTH = 34      # characters per narration line before wrapping


def money(x):
    return "" if x is None else f"{x:,.2f}"


def draw_pdf(rows, path, start_text="01/09/2026", end_text="31/10/2026"):
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4

    def header(y):
        c.setFont("Helvetica-Bold", 7)
        c.rect(COLS[0][1], y - 14, COLS[-1][2] - COLS[0][1], 18)
        for name, x0, x1 in COLS:
            c.drawCentredString((x0 + x1) / 2, y - 8, name)
        return y - 14

    def new_page(first=False):
        if not first:
            c.showPage()
        y = H - 40
        c.setFont("Helvetica-Bold", 11)
        c.drawString(30, y, "FAKE BANK LTD  (synthetic data for testing)")
        c.setFont("Helvetica", 8)
        y -= 16
        c.drawString(30, y, f"Statement From : {start_text}        To : {end_text}")
        y -= 18
        return header(y)

    def vertical_lines(top, bottom):
        for _, x0, _ in COLS:
            c.line(x0, top, x0, bottom)
        c.line(COLS[-1][2], top, COLS[-1][2], bottom)

    y = new_page(first=True)
    table_top = y
    c.setFont("Helvetica", 7)

    for r in rows:
        narr_lines = textwrap.wrap(r["narration"], NARR_WIDTH, break_on_hyphens=False) or [""]
        needed = LINE_H * len(narr_lines) + 4
        if y - needed < 40:                     # page full -> new page
            vertical_lines(table_top, y)
            c.line(COLS[0][1], y, COLS[-1][2], y)
            y = new_page()
            table_top = y
            c.setFont("Helvetica", 7)
        y -= LINE_H
        c.drawString(33, y, r["date"])
        c.drawRightString(342, y, r["ref_no"])
        c.drawString(349, y, r["value_date"])
        c.drawRightString(452, y, money(r["withdrawal"]))
        c.drawRightString(512, y, money(r["deposit"]))
        c.drawRightString(572, y, money(r["closing_balance"]))
        for i, line in enumerate(narr_lines):      # wrapped narration lines
            c.drawString(78, y - i * LINE_H, line)
        y -= LINE_H * (len(narr_lines) - 1) + 4

    vertical_lines(table_top, y)
    c.line(COLS[0][1], y, COLS[-1][2], y)

    # Statement summary, like real HDFC statements (lets the parser verify counts + totals)
    if y < 110:
        c.showPage()
        y = H - 40
    dr = [r for r in rows if r["withdrawal"] is not None]
    cr = [r for r in rows if r["deposit"] is not None]
    opening = rows[0]["closing_balance"] + (rows[0]["withdrawal"] or 0) - (rows[0]["deposit"] or 0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(30, y - 30, "STATEMENT SUMMARY :-")
    c.setFont("Helvetica-Bold", 8)
    for x, label in [(60, "Opening Balance"), (190, "Dr Count"), (260, "Cr Count"),
                     (340, "Debits"), (430, "Credits"), (520, "Closing Bal")]:
        c.drawCentredString(x, y - 46, label)
    c.setFont("Helvetica", 8)
    for x, v in [(60, money(opening)), (190, str(len(dr))), (260, str(len(cr))),
                 (340, money(sum(r["withdrawal"] for r in dr))), (430, money(sum(r["deposit"] for r in cr))),
                 (520, money(rows[-1]["closing_balance"]))]:
        c.drawCentredString(x, y - 58, v)
    c.save()


if __name__ == "__main__":
    rows = generate_transactions()
    draw_pdf(rows, "data/fake_statement.pdf")
    pd.DataFrame(rows).to_csv("data/fake_statement_truth.csv", index=False)
    print(f"Created data/fake_statement.pdf with {len(rows)} transactions")
    print("Created data/fake_statement_truth.csv (the correct answers)")
