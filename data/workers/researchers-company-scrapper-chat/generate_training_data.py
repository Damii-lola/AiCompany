"""
Generates training_examples.csv (context, reply) for the
researchers-company-scrapper-chat worker.

This is checked into the repo alongside its output on purpose: the CSV is
the real training data, but this script is what makes it reviewable/
regenerable/extensible — add a new phrasing pattern or category here
rather than hand-editing hundreds of CSV rows. Every `reply` is a strict
JSON object matching workers/researchers-company-scrapper-chat/
SYSTEM_PROMPT.txt's schema exactly (same field set, same intent values) —
if that file's schema changes, this generator must change with it, then
regenerate the CSV and retrain.

Design principle: this worker must GENERALIZE, not memorize strings — so
every (context) includes realistic noise (typos, filler, "near me",
unsupported qualifiers, multi-turn follow-ups referencing prior context)
across many DIFFERENT categories/addresses/combinations, rather than
being hundreds of near-duplicates of one pattern.
"""
import csv
import json
import os
import random

random.seed(42)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_examples.csv")

CATEGORIES = [
    "coffee shop", "pharmacy", "auto repair shop", "bakery", "dentist",
    "hair salon", "pet grooming", "laundromat", "gym", "nail salon",
    "pizza restaurant", "hardware store", "veterinary clinic", "florist",
    "bookstore", "car wash", "tattoo parlor", "yoga studio",
]

# (typo spelling, correct spelling) pairs reused across templates
TYPOS = {
    "coffee shop": ["Coffe shop", "coffe shops", "cofee shop"],
    "pharmacy": ["pharmacys", "farmacy", "pharmacies"],
    "veterinary clinic": ["vetrinary clinic", "veterinary clinics"],
    "bakery": ["bakerys", "bakery's"],
}

ADDRESSES = [
    "Columbus, OH", "Austin, TX", "123 Main St, Denver, CO",
    "Brooklyn, NY", "Seattle, WA", "Phoenix, AZ", "Miami, FL",
    "Chicago, IL", "Portland, OR", "San Diego, CA",
]

UNSUPPORTED_QUALIFIERS = [
    ("eligible for loans", "loan eligibility"),
    ("with good reviews", "review scores"),
    ("that are hiring", "hiring status"),
    ("open on Sundays", "opening hours"),
    ("making over 500k a year", "revenue"),
    ("with more than 10 employees", "employee count"),
]

rows = []

def add(context, reply_obj):
    rows.append({"context": context, "reply": json.dumps(reply_obj, ensure_ascii=False)})

# ---- 1. Plain search requests, with/without typos, across categories/addresses ----
for cat in CATEGORIES:
    addr = random.choice(ADDRESSES)
    radius = random.choice([1, 2, 3, 5, None])
    phrasing = random.choice([
        f"{cat}s within {radius or 3} miles of {addr}",
        f"show me {cat}s near {addr}",
        f"find {cat}s around {addr}" + (f" within {radius} miles" if radius else ""),
        f"I need a list of {cat}s in {addr}",
    ])
    add(phrasing, {
        "intent": "search", "category_phrase": cat, "address": addr,
        "radius_miles": radius, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })

# ---- 2. Typo'd category phrasing ----
for cat, typo_list in TYPOS.items():
    for typo in typo_list:
        addr = random.choice(ADDRESSES)
        add(f"any {typo} near {addr}?", {
            "intent": "search", "category_phrase": cat, "address": addr,
            "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
            "unsupported_note": None, "clarifying_reply": None,
        })

# ---- 3. No-website-only filter ----
for cat in CATEGORIES[:10]:
    addr = random.choice(ADDRESSES)
    add(f"{cat}s with no website within 2 miles of {addr}", {
        "intent": "search", "category_phrase": cat, "address": addr,
        "radius_miles": 2, "no_website_only": True, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })
    add(f"which {cat}s near {addr} don't have a website", {
        "intent": "search", "category_phrase": cat, "address": addr,
        "radius_miles": None, "no_website_only": True, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })

# ---- 4. Delivery format requests ----
for cat in CATEGORIES[:8]:
    addr = random.choice(ADDRESSES)
    add(f"give me a csv of {cat}s within 3 miles of {addr}", {
        "intent": "search", "category_phrase": cat, "address": addr,
        "radius_miles": 3, "no_website_only": False, "delivery_mode": "csv",
        "unsupported_note": None, "clarifying_reply": None,
    })
    add(f"send me a pdf report of {cat}s near {addr}", {
        "intent": "search", "category_phrase": cat, "address": addr,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "pdf",
        "unsupported_note": None, "clarifying_reply": None,
    })

# ---- 5. Pure count queries, no location, with typo tolerance ----
for cat in CATEGORIES:
    typo = random.choice(TYPOS.get(cat, [cat]))
    add(f"how many {cat}s are in your database", {
        "intent": "count", "category_phrase": cat, "address": None,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })
    add(f"Number of {typo} in ur database", {
        "intent": "count", "category_phrase": cat, "address": None,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })
add("how many businesses are in the database", {
    "intent": "count", "category_phrase": None, "address": None,
    "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
    "unsupported_note": None, "clarifying_reply": None,
})
add("total count of everything you have", {
    "intent": "count", "category_phrase": None, "address": None,
    "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
    "unsupported_note": None, "clarifying_reply": None,
})

# ---- 6. "near me" with no prior context -> must clarify, never geocode literally ----
for cat in CATEGORIES[:10]:
    typo_variants = [cat] + TYPOS.get(cat, [])
    phr = random.choice(typo_variants)
    add(f"Number 9f {phr} near me", {
        "intent": "clarify", "category_phrase": cat, "address": None,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None,
        "clarifying_reply": "I don't have access to your location — what city or address should I search near?",
    })
    add(f"{phr} near me", {
        "intent": "clarify", "category_phrase": cat, "address": None,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None,
        "clarifying_reply": "I don't have access to your location — what city or address should I search near?",
    })

# ---- 7. Unsupported qualifiers: still answer the supported part, disclose the rest ----
for cat in CATEGORIES[:10]:
    qualifier, label = random.choice(UNSUPPORTED_QUALIFIERS)
    add(f"Number of {cat}s {qualifier}", {
        "intent": "count", "category_phrase": cat, "address": None,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": f"I don't have {label} data, so that part of your question wasn't applied.",
        "clarifying_reply": None,
    })
for cat in CATEGORIES[8:16]:
    qualifier, label = random.choice(UNSUPPORTED_QUALIFIERS)
    addr = random.choice(ADDRESSES)
    add(f"{cat}s near {addr} {qualifier}", {
        "intent": "search", "category_phrase": cat, "address": addr,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": f"I don't have {label} data, so that part of your question wasn't applied.",
        "clarifying_reply": None,
    })

# ---- 8. No location at all, non-count search request -> clarify ----
for cat in CATEGORIES[:8]:
    add(f"find me some {cat}s", {
        "intent": "clarify", "category_phrase": cat, "address": None,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None,
        "clarifying_reply": "Sure — what city or address should I search near?",
    })

# ---- 9. Multi-turn follow-ups: context carryover ----
# Each of these includes the PRIOR context as part of the training context
# string, formatted exactly like render-service will send it: a small JSON
# snapshot followed by the new message.
def ctx_prefix(prior):
    return "Prior context: " + json.dumps(prior, ensure_ascii=False) + "\nNew message: "

for cat in CATEGORIES[:9]:
    addr = random.choice(ADDRESSES)
    prior = {"category_phrase": cat, "address": addr, "radius_miles": 3, "no_website_only": False, "delivery_mode": "chat"}
    # a) contentless follow-up -> carry everything over, no address change
    add(ctx_prefix(prior) + "list all of them", {
        "intent": "search", "category_phrase": cat, "address": None,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })
    # b) switches only the filter -> carry category/address/radius over
    add(ctx_prefix(prior) + "now only show ones with no website", {
        "intent": "search", "category_phrase": None, "address": None,
        "radius_miles": None, "no_website_only": True, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })
    # c) switches only delivery format -> carry everything else over
    add(ctx_prefix(prior) + "give me that as a csv", {
        "intent": "search", "category_phrase": None, "address": None,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "csv",
        "unsupported_note": None, "clarifying_reply": None,
    })
    # d) names a brand-new address -> treat as fresh, drop old context entirely
    new_addr = random.choice([a for a in ADDRESSES if a != addr])
    add(ctx_prefix(prior) + f"what about near {new_addr} instead", {
        "intent": "search", "category_phrase": None, "address": new_addr,
        "radius_miles": None, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })
    # e) widens the radius only
    add(ctx_prefix(prior) + "widen that to 10 miles", {
        "intent": "search", "category_phrase": None, "address": None,
        "radius_miles": 10, "no_website_only": False, "delivery_mode": "chat",
        "unsupported_note": None, "clarifying_reply": None,
    })

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["context", "reply"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} training examples to {OUT_PATH}")
