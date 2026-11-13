"""
generate_training_data.py
-------------------------
Creates LABELLED examples for the categoriser:
    narration (input)  ->  category (correct answer / "label")

We write narrations in the REAL HDFC formats we discovered on Day 1
(UPI-, POS, ACH C-/D-, TPT, /PAYU..., ATW-, REF-, NEFT CR, IMPS).

THE MOST IMPORTANT IDEA TODAY -- three separate files:
  train.csv        -> the model LEARNS from these
  test_seen.csv    -> NEW examples, but of merchants that also appear in train
  test_unseen.csv  -> merchants/shops the model has NEVER seen in training

test_unseen is the real exam: in real life, people pay shops
that weren't in anyone's training data.

Run:  python generate_training_data.py
"""

import random

import pandas as pd

random.seed(7)

# ---------------------------------------------------------------
# 1. Merchants. Big brands appear under their BRAND name and sometimes
#    their LEGAL company name (that's what banks often print!)
# ---------------------------------------------------------------
BRANDS = {
    "Food": ["SWIGGY", "BUNDL TECHNOLOGIES", "ZOMATO", "ZOMATO LIMITED", "DOMINOS PIZZA",
             "JUBILANT FOODWORKS", "MCDONALDS", "KFC", "STARBUCKS", "TATA STARBUCKS",
             "BURGER KING", "HALDIRAMS", "A2B ADYAR ANANDA BHAVAN", "SARAVANA BHAVAN", "EATSURE"],
    "Groceries": ["BLINKIT", "BLINK COMMERCE", "ZEPTO", "ZEPTO MARKETPLACE PR", "KIRANAKART TECHNOLOGIES",
                  "BIGBASKET", "INNOVATIVE RETAIL CONCEPTS", "DMART", "AVENUE SUPERMARTS",
                  "RELIANCE SMART", "MORE RETAIL", "JIOMART", "SWIGGY INSTAMART", "NATURES BASKET"],
    "Shopping": ["AMAZON", "AMAZON SELLER SERVICES", "FLIPKART", "FLIPKART INTERNET", "MYNTRA",
                 "MYNTRA DESIGNS", "AJIO", "NYKAA", "FSN E-COMMERCE", "MEESHO", "CROMA",
                 "RELIANCE DIGITAL", "DECATHLON", "LIFESTYLE", "WESTSIDE", "IKEA"],
    "Travel": ["UBER", "UBER INDIA SYSTEMS", "OLA", "ANI TECHNOLOGIES", "RAPIDO",
               "ROPPEN TRANSPORTATION", "IRCTC", "REDBUS", "MAKEMYTRIP", "INDIGO",
               "INTERGLOBE AVIATION", "AIR INDIA", "FASTAG", "NAMMA METRO", "CHENNAI METRO RAIL"],
    "Fuel": ["INDIAN OIL", "IOCL", "HPCL", "HINDUSTAN PETROLEUM", "BHARAT PETROLEUM", "BPCL",
             "SHELL INDIA", "NAYARA ENERGY", "JIO BP"],
    "Bills": ["JIO PREPAID", "RELIANCE JIO", "AIRTEL", "BHARTI AIRTEL", "VI VODAFONE IDEA",
              "BESCOM", "TNEB", "TANGEDCO", "TATA PLAY", "ACT FIBERNET", "INDANE GAS",
              "LIC OF INDIA", "HDFC LIFE", "BAJAJ FINANCE", "MAHANAGAR GAS"],
    "Subscriptions": ["NETFLIX", "SPOTIFY", "YOUTUBE PREMIUM", "GOOGLE PLAY", "APPLE MEDIA SERVICES",
                      "DISNEY HOTSTAR", "NOVI DIGITAL", "AMAZON PRIME", "SONYLIV", "ZEE5",
                      "LINKEDIN PREMIUM", "MICROSOFT 365", "CHATGPT OPENAI"],
    "Health": ["APOLLO PHARMACY", "APOLLO HOSPITALS", "MEDPLUS", "NETMEDS", "TATA 1MG", "PHARMEASY",
               "PRACTO", "THYROCARE", "DR LAL PATHLABS", "CULT FIT", "CUREFIT", "FORTIS HOSPITAL"],
    "Entertainment": ["BOOKMYSHOW", "BIG TREE ENTERTAINMENT", "PVR INOX", "PVR LIMITED", "DISTRICT",
                      "ASTROTALK SERVICES P", "DREAM11", "SPORTA TECHNOLOGIES", "STEAM GAMES",
                      "PAYTM INSIDER", "WONDERLA", "SMAAAASH"],
    "Education": ["UDEMY", "COURSERA", "BYJUS", "THINK AND LEARN", "UNACADEMY", "SORTING HAT",
                  "PHYSICSWALLAH", "VEDANTU", "UPGRAD", "SCALER ACADEMY", "SHIV NADAR UNIVERSITY",
                  "ANNA UNIVERSITY", "HIGGINBOTHAMS"],
    "Investments": ["ZERODHA", "ZERODHA BROKING", "GROWW", "NEXTBILLION TECHNOLOGY", "UPSTOX",
                    "ICICI PRUDENTIAL MF", "SBI MUTUAL FUND", "HDFC MUTUAL FUND", "NIPPON INDIA MF",
                    "PARAG PARIKH MF", "ANGEL ONE", "COIN BY ZERODHA"],
}

# Local shops: <name> + <shop word>. The SHOP WORD carries the meaning.
SHOP_WORDS = {
    "Food": ["HOTEL", "BHAVAN", "CAFE", "BAKERY", "SWEETS", "RESTAURANT", "BIRYANI CENTRE",
             "FAST FOOD", "TEA STALL", "JUICE CENTRE", "MESS", "DHABA", "TIFFIN CENTRE", "FOOD COURT"],
    "Groceries": ["STORES", "PROVISION STORE", "SUPERMARKET", "TRADERS", "KIRANA", "VEGETABLES",
                  "FRUITS", "DEPARTMENTAL STORE", "MILK DAIRY", "AGENCIES"],
    "Shopping": ["TEXTILES", "FASHIONS", "SILKS", "FOOTWEAR", "MOBILES", "ELECTRONICS",
                 "GIFT CENTRE", "JEWELLERS", "BOUTIQUE", "OPTICALS"],
    "Travel": ["TRAVELS", "TOURS AND TRAVELS", "PARKING", "CAB SERVICE", "TOLL PLAZA", "BUS SERVICE"],
    "Fuel": ["FUELS", "PETROLEUM", "FILLING STATION", "SERVICE STATION", "PETRO POINT"],
    "Health": ["MEDICALS", "PHARMACY", "CLINIC", "HOSPITAL", "DIAGNOSTICS", "LABS",
               "DENTAL CARE", "EYE CARE", "NURSING HOME"],
    "Education": ["ACADEMY", "TUITION CENTRE", "COACHING CENTRE", "MATRIC SCHOOL", "BOOK HOUSE",
                  "COLLEGE", "INSTITUTE"],
    "Entertainment": ["CINEMAS", "THEATRE", "GAMING ZONE", "BOWLING", "AMUSEMENT PARK"],
    "Bills": ["GAS AGENCY", "CABLE TV", "BROADBAND"],
}
SHOP_NAMES = ["SRI MURUGAN", "LAKSHMI", "GANESH", "BALAJI", "ANNAPOORNA", "SARAVANA", "KRISHNA",
              "VINAYAKA", "SAI", "ROYAL", "NEW STAR", "GREEN PARK", "CITY", "ANAND", "MAHALAKSHMI",
              "SHREE", "JAI HIND", "KAMAKSHI", "SELVAM", "AMMAN", "RAJ", "MEENAKSHI", "VELAN",
              "SRI RAM", "KUMARAN", "PONNI", "ARUL", "THANGAM", "DEVI", "MAGESH"]

PEOPLE = ["RAHUL SHARMA", "PRIYA NAIR", "ANIL KUMAR", "SUNITA DEVI", "VIKRAM SINGH", "MEERA IYER",
          "ARJUN REDDY", "KAVYA MENON", "S KARTHIK", "R LAKSHMI", "DEEPAK JOSHI", "FATIMA BEGUM",
          "JOHN MATHEW", "K SURESH", "ANJALI GUPTA", "M SELVI", "RAVI SHANKAR", "POOJA PATEL",
          "ABDUL RAHMAN", "N PRAKASH", "SNEHA KULKARNI", "GURPREET KAUR", "T MURUGAN", "DIVYA RAO"]

EMPLOYERS = ["ACME TECHNOLOGIES PVT LTD", "INFOSYS LIMITED", "TATA CONSULTANCY SERVICES",
             "ZOHO CORPORATION", "WIPRO LIMITED", "FRESHWORKS TECHNOLOGIES", "HCL TECHNOLOGIES"]
DIVIDEND_COS = ["BATA", "RBL BANK", "ITC", "INFOSYS", "HDFC BANK", "COAL INDIA", "TATA STEEL"]

BANKS = ["SBIN", "ICIC", "UTIB", "HDFC", "KKBK", "IOBA", "YESB", "PUNB"]
HANDLES = ["ybl", "okaxis", "okicici", "oksbi", "paytm", "axl", "ibl", "hdfcbank", "icici"]
CITIES = ["CHENNAI", "BANGALORE", "NAGERCOIL", "MADURAI", "COIMBATORE", "HYDERABAD", "MUMBAI"]


# ---------------------------------------------------------------
# 2. HELD-OUT split: some merchants / names / shop words NEVER go in train
# ---------------------------------------------------------------
def split(items, frac):
    items = items[:]
    random.shuffle(items)
    k = max(1, round(len(items) * frac))
    return items[k:], items[:k]          # (train part, held-out part)


brand_tr, brand_te = {}, {}
for cat, lst in BRANDS.items():
    brand_tr[cat], brand_te[cat] = split(lst, 0.3)
word_tr, word_te = {}, {}
for cat, lst in SHOP_WORDS.items():
    word_tr[cat], word_te[cat] = split(lst, 0.3) if len(lst) > 3 else (lst, lst)
name_tr, name_te = split(SHOP_NAMES, 0.3)
people_tr, people_te = split(PEOPLE, 0.3)


# ---------------------------------------------------------------
# 3. Narration formats (copied from the real HDFC statement)
# ---------------------------------------------------------------
def digits(n):
    return "".join(random.choice("0123456789") for _ in range(n))


def code(n):
    return "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(n))


def handle_for(name):
    base = "".join(ch for ch in name.lower() if ch.isalnum())[:12] or "merchant"
    return f"{base}{random.choice(['', '.pay', digits(2), '.rzp'])}@{random.choice(HANDLES)}".upper()


def upi(name, credit=False):
    # HDFC cuts UPI names to ~20 characters, e.g. "ASTROTALK SERVICES P"
    note = random.choice(["UPI", "PAYMENT", "PAY TO MERCHANT", "ORDER", "COLLECT", "UPI INTENT"])
    s = "S " if credit else random.choice(["", ""])
    return f"UPI-{s}{name[:20]}-{handle_for(name)}-{random.choice(BANKS)}0{digits(6)}-{digits(12)}-{note}"


def pos(name):
    return f"POS 512967XXXXXX{digits(4)} {name[:23]} {random.choice(CITIES)}"


def gateway(name):
    gw = random.choice(["PAYU", "RAZORPAY", "BILLDESK", "CCAVENUE"])
    return f"{code(18)}/{gw}{name.replace(' ', '')}"


def ach_d(name, cat="Bills"):
    word = {"Investments": ["SIP", "MF SIP"], "Subscriptions": ["MANDATE", "AUTOPAY"]}.get(cat, ["EMI", "NACH", "MANDATE"])
    return f"ACH D- {name} {random.choice(word)}-{digits(6)}"


def merchant_narration(cat, name):
    """Pick a realistic format for this kind of merchant."""
    if cat == "Investments":
        f = random.choice([upi, ach_d, gateway])
    elif cat in ("Bills", "Subscriptions"):
        f = random.choice([upi, upi, gateway, ach_d, pos])
    else:
        f = random.choice([upi, upi, upi, pos, gateway])
    return f(name, cat) if f is ach_d else f(name)


def special(cat, people):
    """Transfers, Cash, Income, Refund -- formats that don't involve a shop."""
    p = random.choice(people)
    if cat == "Transfers":
        return random.choice([
            lambda: upi(p, credit=random.random() < 0.5),
            lambda: f"FT-XX{digits(4)}-IMPS TRANSACTION-{p}-{random.choice('FGH')}{digits(11)}",
            lambda: f"{digits(14)}-TPT-{code(16)}-{p}",
            lambda: f"NEFT DR-{random.choice(BANKS)}0{digits(6)}-{p}-NETBANK-{digits(12)}",
        ])()
    if cat == "Cash":
        return f"ATW-512967XXXXXX{digits(4)}-S1A{code(5)}-{random.choice(CITIES)}"
    if cat == "Income":
        return random.choice([
            lambda: f"NEFT CR-CITI0000002-{random.choice(EMPLOYERS)}-SALARY {random.choice(['SEP', 'OCT', 'NOV'])} 2026-{digits(12)}",
            lambda: f"ACH C- {random.choice(DIVIDEND_COS)} {random.choice(['INTDIV', 'FINDIV', 'DIV'])} {digits(2)} {digits(2)}-{digits(6)}",
            lambda: "CREDIT INTEREST CAPITALISED",
            lambda: f"UPI-S CASHBACK-{random.choice(['GPAY', 'PHONEPE', 'PAYTM'])}REWARDS@{random.choice(HANDLES).upper()}-{random.choice(BANKS)}0{digits(6)}-{digits(12)}-CASHBACK",
        ])()
    if cat == "Refund":
        return random.choice([
            lambda: f"REF-{random.choice(['PAYU', 'RZP', 'BDSK'])}{random.choice(['IRC', 'AMZ', 'FLP', 'SWG', 'ZMT'])}-{code(18)}-{random.randint(1, 28)}/{random.randint(1, 12)}/26",
            lambda: f"UPI-S {random.choice(['AMAZON', 'FLIPKART', 'SWIGGY', 'ZOMATO', 'IRCTC'])}-REFUND-{random.choice(BANKS)}0{digits(6)}-{digits(12)}-REFUND",
            lambda: f"REV-UPI-{digits(12)}-{random.choice(['FAILED TXN', 'REVERSAL'])}",
        ])()
    raise ValueError(cat)


# ---------------------------------------------------------------
# 4. Build one example
# ---------------------------------------------------------------
SPECIAL = ["Transfers", "Cash", "Income", "Refund"]
# rough real-life frequency of each category
WEIGHTS = {"Food": 14, "Groceries": 10, "Shopping": 9, "Travel": 8, "Fuel": 4, "Bills": 7,
           "Subscriptions": 4, "Health": 5, "Entertainment": 4, "Education": 3, "Investments": 4,
           "Income": 6, "Transfers": 12, "Cash": 4, "Refund": 3}


def example(seen=True):
    cat = random.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
    if cat in SPECIAL:
        return special(cat, people_tr if seen else people_te), cat
    brands = brand_tr if seen else brand_te
    words = word_tr if seen else word_te
    names = name_tr if seen else name_te
    if cat in words and random.random() < 0.45:          # local shop
        name = f"{random.choice(names)} {random.choice(words[cat])}"
    else:                                                # big brand
        name = random.choice(brands[cat])
    return merchant_narration(cat, name), cat


def build(n, seen):
    rows, seen_texts = [], set()
    while len(rows) < n:
        narr, cat = example(seen)
        if narr not in seen_texts:
            seen_texts.add(narr)
            rows.append({"narration": narr, "category": cat})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    train = build(2000, seen=True)
    test_seen = build(300, seen=True)
    test_unseen = build(300, seen=False)
    train.to_csv("data/train.csv", index=False)
    test_seen.to_csv("data/test_seen.csv", index=False)
    test_unseen.to_csv("data/test_unseen.csv", index=False)

    print(f"train.csv: {len(train)}   test_seen.csv: {len(test_seen)}   test_unseen.csv: {len(test_unseen)}\n")
    print("Category counts in train:")
    print(train.category.value_counts().to_string(), "\n")
    print("Examples:")
    for _, r in train.sample(8, random_state=1).iterrows():
        print(f"  [{r.category:13s}] {r.narration}")
    print("\nNever-seen merchants in test_unseen (sample):",
          ", ".join(random.sample(sum(brand_te.values(), []), 6)))
