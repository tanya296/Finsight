"""
parse_statement.py
------------------
Reads an HDFC-style bank statement PDF and turns it into a clean table,
then saves it into a SQLite database (a database that is just one file).

The tricky part: long narrations WRAP onto 2-3 lines in the PDF.
A naive reader thinks each line is a new transaction. We fix that by
using a simple rule:

    A line that has a DATE in the Date column  -> starts a NEW transaction
    A line WITHOUT a date                      -> continuation of the previous
                                                  transaction's narration

Run:  python parse_statement.py data/fake_statement.pdf
Output: data/transactions.db  (SQLite)  and  data/transactions.csv
"""

import re
import sqlite3
import sys

import pandas as pd
import pdfplumber

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{2}$")          # e.g. 01/09/26

# Keyword in a header label -> our clean column name.
# We match single KEYWORDS (not full labels like "Withdrawal Amt.") because
# different PDFs split header text into words differently.
HEADER_KEYWORDS = {
    "narration": "narration",
    "chq": "ref_no", "ref": "ref_no",
    "value": "value_date",
    "withdrawal": "withdrawal", "debit": "withdrawal",
    "deposit": "deposit", "credit": "deposit",
    "balance": "closing_balance",
    "date": "date",
}
NEEDED = {"date", "narration", "ref_no", "value_date", "withdrawal", "deposit", "closing_balance"}

# Lines after the table (summary, footer) start with these -> stop reading page
STOP_WORDS = ("STATEMENT SUMMARY", "OPENING BALANCE", "GENERATED ON", "PAGE NO",
              "THIS IS A COMPUTER", "*CLOSING BALANCE", "HDFC BANK LIMITED")

DEBUG = "--debug" in sys.argv   # prints LAYOUT only, never transaction data


def keyword_of(text):
    t = re.sub(r"[^a-z]", " ", text.lower()).split()
    for word in t:
        if word in HEADER_KEYWORDS:
            return HEADER_KEYWORDS[word]
    return None


# ---------------------------------------------------------------
# Step 1: find where each column is on the page
# ---------------------------------------------------------------
def find_columns(page):
    """
    Look for the header row ("Date", "Narration", ...) and work out the
    x-range (left edge, right edge) of every column.
    Returns (columns, header_bottom_y) or (None, None) if no header found.
    """
    words = page.extract_words(x_tolerance=1.5)
    anchor = next((w for w in words if w["text"].lower().startswith("narration")), None)
    if anchor is None:
        return None, None

    # Header words = words near the "Narration" word vertically
    # (allows 2-line headers like "Withdrawal" / "Amt.")
    band = [w for w in words if abs(w["top"] - anchor["top"]) < 12]
    found = {}
    for w in band:
        name = keyword_of(w["text"])
        if name == "date" and "date" in found:
            continue                       # keep the FIRST "Date" (leftmost later)
        if name and name not in found:
            found[name] = w
    if DEBUG:
        print(f"[debug] header words near 'Narration': {[w['text'] for w in band]}")
        print(f"[debug] columns recognised: {sorted(found)}")
    if not NEEDED.issubset(found):
        if DEBUG:
            print(f"[debug] MISSING columns: {sorted(NEEDED - set(found))}")
        return None, None

    headers = sorted((w["x0"], w["x1"], name, w["bottom"]) for name, w in found.items())
    header_bottom = max(w["bottom"] for w in band)

    # BEST: use the table's vertical border lines as column edges.
    # (Header labels are CENTRED, so they don't tell us where a column starts.)
    # page.edges includes lines AND the sides of rectangles.
    xs = sorted({round(e["x0"]) for e in page.edges
                 if e.get("orientation") == "v" and e["bottom"] > header_bottom})
    if DEBUG:
        print(f"[debug] vertical border x-positions: {xs}")
    if len(xs) >= len(headers) + 1 and all(
            any(x <= (h[0] + h[1]) / 2 for x in xs) and any(x > (h[0] + h[1]) / 2 for x in xs)
            for h in headers):
        cols = []
        for x0, x1, name, _ in headers:
            mid = (x0 + x1) / 2
            left = max(x for x in xs if x <= mid)
            right = min(x for x in xs if x > mid)
            cols.append((name, left, right))
        return cols, header_bottom

    # FALLBACK (no border lines): halfway between neighbouring header labels
    cols = []
    for i, (x0, x1, name, _) in enumerate(headers):
        left = 0 if i == 0 else (headers[i - 1][1] + x0) / 2
        right = page.width if i == len(headers) - 1 else (x1 + headers[i + 1][0]) / 2
        cols.append((name, left, right))
    return cols, header_bottom


# ---------------------------------------------------------------
# Step 2: group words into lines, and lines into cells
# ---------------------------------------------------------------
def page_lines(page, cols, header_bottom):
    """Return a list of dicts, one per visual line: {column_name: text}."""
    words = page.extract_words(x_tolerance=1.5, y_tolerance=2)
    words = [w for w in words if w["top"] > header_bottom + 1]

    # Group words that sit on the same horizontal line (similar 'top' value)
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) < 3:
            lines[-1].append(w)
        else:
            lines.append([w])

    result = []
    seen_txn = False                 # have we passed a transaction line on this page yet?
    for line in lines:
        text = " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"])).upper()
        if text.startswith(STOP_WORDS):
            if seen_txn:
                if DEBUG:
                    print(f"[debug] stopped at footer word '{next(s for s in STOP_WORDS if text.startswith(s))}'")
                break                # footer/summary AFTER the table -> end of this page
            continue                 # same words in the page HEADER (top) -> just skip the line
        cells = {}
        for w in sorted(line, key=lambda w: w["x0"]):
            mid = (w["x0"] + w["x1"]) / 2          # which column is the word in?
            for name, left, right in cols:
                if left <= mid < right:
                    cells[name] = (cells.get(name, "") + " " + w["text"]).strip()
                    break
        if DATE_RE.match(cells.get("date", "")):
            seen_txn = True
        result.append(cells)
    return result


# ---------------------------------------------------------------
# Step 3: merge wrapped lines into complete transactions
# ---------------------------------------------------------------
def to_number(s):
    """'1,23,456.78' -> 123456.78 ; '' -> None"""
    if not isinstance(s, str):      # empty cells come through as NaN
        return None
    s = s.replace(",", "").strip()
    return float(s) if s else None


def merge_lines(all_lines):
    txns = []
    pieces = []  # narration pieces of the current transaction
    for cells in all_lines:
        if DATE_RE.match(cells.get("date", "")):
            if txns:
                txns[-1]["narration_pieces"] = pieces
            txns.append(dict(cells))
            pieces = [cells.get("narration", "")]
        elif txns and cells.get("narration"):
            pieces.append(cells["narration"])     # continuation line
        # anything else (page titles, footers) is ignored
    if txns:
        txns[-1]["narration_pieces"] = pieces

    # How wide is a "full" narration line? Use the longest one we saw.
    width = max((len(p) for t in txns for p in t["narration_pieces"]), default=0)

    for t in txns:
        # If a line is FULL width, the bank cut the text mid-word -> join with "".
        # If it's SHORTER, the bank wrapped at a space -> join with " ".
        text = ""
        for p in t.pop("narration_pieces"):
            if text:
                # (Must be EXACTLY full: a line 1 char short means the bank
                #  wrapped at a space, e.g. "...-RAVI" / "KUMAR")
                text += "" if len(prev) >= width else " "
            text += p
            prev = p
        t["narration"] = text
    return txns


# ---------------------------------------------------------------
# Step 4: understand the narration (type + who)
# ---------------------------------------------------------------
def classify(narration):
    """
    Pull out the transaction TYPE and the COUNTERPARTY (who the money
    went to / came from). These are simple rules (regex) -- on Day 2-3
    we'll compare them against an AI model.
    """
    n = narration.upper()
    if n.startswith("UPI-"):
        # Format: UPI-<NAME>-<UPI ID>-<BANK CODE>-<REF>-<NOTE>
        body = n[4:]
        if body.startswith("S "):          # 'UPI-S' variant seen in real HDFC
            body = body[2:]
        return "UPI", body.split("-")[0].strip()
    if "IMPS" in n:
        who = n.split("IMPS TRANSACTION-")[-1] if "IMPS TRANSACTION-" in n else ""
        # Real HDFC adds "-<reference no.>" at the end, e.g. "NAME-G51239876543"
        who = re.sub(r"-[A-Z]?\d{6,}.*$", "", who)
        return "IMPS", who.strip()
    if n.startswith("NEFT"):
        parts = n.split("-")
        return "NEFT", parts[2].strip() if len(parts) > 2 else ""
    if n.startswith("ATW") or "ATM" in n:
        return "ATM", "ATM"
    if n.startswith("POS"):
        m = re.match(r"POS \S+ (.+?) [A-Z]+$", n)
        return "CARD", m.group(1) if m else ""
    if "INTEREST" in n:
        return "INTEREST", "BANK"

    # --- formats discovered in a REAL HDFC statement ---
    # ACH = automatic payment. "ACH C-" = money in (e.g. dividends), "ACH D-" = out (e.g. EMI)
    m = re.match(r"ACH ([CD])-\s*(.*)", n)
    if m:
        who = []
        for tok in m.group(2).split():
            if re.search(r"\d", tok) or tok in ("INTDIV", "FINDIV", "DIV", "DIVIDEND"):
                break                      # stop at codes / numbers
            who.append(tok)
        return "ACH", " ".join(who)
    # TPT = Third Party Transfer: "<account no>-TPT-<ref>-<NAME>"
    if re.match(r"^\d{8,}-TPT-", n):
        return "TPT", n.split("-")[-1].strip()
    # Refund through a payment gateway: "REF-PAYU..."
    if n.startswith("REF-"):
        return "REFUND", ""
    # Payment via gateway: "<code>/PAYU<MERCHANT>" e.g. ".../PAYUZOMATOLIMITED"
    m = re.search(r"/(PAYU|RAZORPAY|CCAVENUE|BILLDESK)?([A-Z ]+)$", n)
    if m:
        who = re.sub(r"(LIMITED|LTD|PVT|PRIVATE)$", "", m.group(2)).strip()
        return "GATEWAY", who
    return "OTHER", ""


# ---------------------------------------------------------------
# Main: PDF -> table -> SQLite
# ---------------------------------------------------------------
def open_pdf(path, password=None):
    """
    Open a PDF. Bank e-statements are usually password-protected, so if
    opening fails we ASK for the password. getpass hides what you type, and
    the password is never saved in the code, a file, or terminal history.
    """
    try:
        pdf = pdfplumber.open(path, password=password or "")
        pdf.pages[0].extract_words()     # forces decryption -> fails if locked
        return pdf
    except Exception:
        if password is not None:
            raise ValueError("Wrong PDF password.")
        from getpass import getpass
        pw = getpass(f"'{path}' is password-protected. Enter password (hidden): ")
        return open_pdf(path, password=pw)


def read_summary(text):
    """
    HDFC statements end with a STATEMENT SUMMARY:
        Opening Balance  Dr Count  Cr Count  Debits      Credits     Closing Bal
        25,000.00        127       13        1,89,711.30 2,02,168.85 37,457.55
    Returns {'dr_count', 'cr_count', 'debits', 'credits'} or None if not found.
    Flexible: PDFs sometimes squash 'Dr Count' into 'DrCount', or split the header line.
    """
    lines = text.upper().splitlines()
    for i, line in enumerate(lines):
        if re.search(r"DR\s*COUNT", line) or re.search(r"CR\s*COUNT", line):
            for nxt in lines[i + 1:i + 5]:
                nums = re.findall(r"\d[\d,]*(?:\.\d+)?", nxt)
                ints = [int(t) for t in nums if "." not in t and "," not in t]
                decs = [float(t.replace(",", "")) for t in nums if "." in t]
                if len(ints) >= 2:
                    out = {"dr_count": ints[0], "cr_count": ints[1]}
                    if len(decs) >= 4:            # opening, debits, credits, closing
                        out["debits"], out["credits"] = decs[1], decs[2]
                    return out
    return None


def parse_pdf(path, password=None):
    all_lines = []
    last_cols = None
    summary = None
    with open_pdf(path, password) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            if DEBUG:
                print(f"\n[debug] --- page {i} ---")
            summary = summary or read_summary(page.extract_text() or "")
            cols, header_bottom = find_columns(page)
            if cols is None:
                if last_cols is None:
                    continue              # no table seen yet (e.g. cover page)
                if DEBUG:
                    print("[debug] no header on this page -> reusing previous page's columns")
                cols, header_bottom = last_cols, 0   # table continues w/o header
                lines = page_lines(page, cols, header_bottom)
                # page has no header, so lines above the first transaction are page-header
                # text (bank name, address...) -> drop them so they don't get glued on
                first = next((k for k, l in enumerate(lines) if DATE_RE.match(l.get("date", ""))), len(lines))
                lines = lines[first:]
            else:
                lines = page_lines(page, cols, header_bottom)
            last_cols = cols
            if DEBUG:
                print(f"[debug] columns (name, left x, right x): "
                      f"{[(n, round(l), round(r)) for n, l, r in cols]}")
                n_dated = sum(bool(DATE_RE.match(l.get('date', ''))) for l in lines)
                print(f"[debug] {len(lines)} text lines, {n_dated} start with a date")
                bad = [l.get("date", "") for l in lines if l.get("date") and not DATE_RE.match(l["date"])]
                # show only the SHAPE of odd date cells (digits->9, letters->A), not content
                shapes = {re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", b)) for b in bad}
                print(f"[debug] odd 'date' cell shapes: {sorted(shapes)[:8]}")
            all_lines += lines

    txns = merge_lines(all_lines)
    if not txns:
        raise SystemExit("No transactions found. Run again with --debug at the end "
                         "and send Claude the [debug] lines (they contain no transaction data).")
    df = pd.DataFrame(txns)
    for c in ["withdrawal", "deposit", "closing_balance"]:
        df[c] = df.get(c, pd.Series(dtype=str)).apply(to_number)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y").dt.strftime("%Y-%m-%d")
    df["value_date"] = pd.to_datetime(df["value_date"], format="%d/%m/%y").dt.strftime("%Y-%m-%d")
    df[["txn_type", "counterparty"]] = df["narration"].apply(lambda n: pd.Series(classify(n)))
    df["amount"] = df["deposit"].fillna(0) - df["withdrawal"].fillna(0)   # +in / -out
    df.attrs["summary"] = summary
    return df[["date", "value_date", "narration", "ref_no", "withdrawal", "deposit",
               "amount", "closing_balance", "txn_type", "counterparty"]]


def sanity_check(df):
    """
    Each balance should equal previous balance + this transaction.
    If not, we probably mis-read a row. A great self-check!
    """
    bad = 0
    for i in range(1, len(df)):
        expected = round(df.closing_balance[i - 1] + df.amount[i], 2)
        if abs(expected - df.closing_balance[i]) > 0.01:
            bad += 1
            print(f"  Balance mismatch on row {i}: {df.narration[i][:40]}")
    print(f"Balance check: {len(df) - 1 - bad}/{len(df) - 1} rows consistent")

    # The balance check can't see rows MISSING at the end -> compare with the bank's own counts
    s = df.attrs.get("summary")
    if s:
        dr, cr = int((df.amount < 0).sum()), int((df.amount > 0).sum())
        ok = (dr, cr) == (s["dr_count"], s["cr_count"])
        print(f"Count check vs statement summary: parsed {dr} withdrawals + {cr} deposits, "
              f"bank says {s['dr_count']} + {s['cr_count']} -> {'OK' if ok else 'MISSING ROWS!'}")
        if "debits" in s:
            d, c = round(df.withdrawal.sum(), 2), round(df.deposit.sum(), 2)
            ok2 = abs(d - s["debits"]) < 0.01 and abs(c - s["credits"]) < 0.01
            print(f"Totals check: parsed debits {d:,.2f} / credits {c:,.2f}, bank says "
                  f"{s['debits']:,.2f} / {s['credits']:,.2f} -> {'OK' if ok2 else 'MISMATCH!'}")
    else:
        print("Count check: no statement summary found (can't verify the total count)")


def save_sqlite(df, db_path="data/transactions.db"):
    with sqlite3.connect(db_path) as con:
        df.to_sql("transactions", con, if_exists="replace", index=False)
    print(f"Saved {len(df)} rows to {db_path} (table: transactions)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check-only" in sys.argv     # verify without overwriting the database
    path = args[0] if args else "data/fake_statement.pdf"
    df = parse_pdf(path)
    print(f"Parsed {len(df)} transactions from {path}\n")
    print(df[["date", "txn_type", "counterparty", "amount"]].head(10).to_string(index=False))
    print()
    sanity_check(df.reset_index(drop=True))
    if check_only:
        print("(--check-only: nothing saved)")
    else:
        df.to_csv("data/transactions.csv", index=False)
        save_sqlite(df)
