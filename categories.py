"""
categories.py -- the list of spending categories, shared by every script.
Keeping it in ONE place means the data, the rules, the LLM and the
fine-tuned model all use exactly the same labels.
"""

CATEGORIES = {
    "Food": "restaurants, cafes, food delivery (Swiggy, Zomato), bakeries, sweets",
    "Groceries": "supermarkets, kirana stores, vegetables, quick-commerce (Blinkit, Zepto)",
    "Shopping": "online/offline shopping: clothes, electronics, Amazon, Flipkart",
    "Travel": "cabs, trains, flights, buses, parking, tolls",
    "Fuel": "petrol pumps, fuel stations",
    "Bills": "electricity, mobile/DTH recharge, gas, broadband, insurance, loan EMI",
    "Subscriptions": "Netflix, Spotify, YouTube Premium and other recurring apps",
    "Health": "pharmacies, hospitals, clinics, diagnostic labs",
    "Entertainment": "movies, events, gaming, astrology/fun apps",
    "Education": "school/college fees, courses, tuition, books",
    "Investments": "mutual fund SIPs, stocks, brokers",
    "Income": "salary, dividends, interest, cashback received",
    "Transfers": "money sent to / received from a person",
    "Cash": "ATM cash withdrawal",
    "Refund": "refunds and reversals",
}

LABELS = list(CATEGORIES)
