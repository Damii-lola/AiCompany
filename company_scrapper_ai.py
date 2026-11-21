"""
Researchers / Company Scrapper — plain-English SBA trade-area research
==============================================================================
First AI in the "Researchers" category (Company Scrapper, Basic Researcher,
Advance Researcher, Organizer). Job: a user types a plain-English research
request, this turns it into a real structured query against the live
database (render-service's API, ~18.8M real US+territory businesses across
however many Prisma shards are configured) and hands back real deliverable
files (CSV/XLSX/PDF) — the actual locked SBA V1 product.

RUNS AS A GITHUB ACTIONS SYSTEM, NOT A KAGGLE GPU NOTEBOOK (changed Nov
2026, on the user's explicit call: "it's a system na, let it run on
GitHub Actions"). This is a real engineering tradeoff, stated plainly:
this worker previously ran Qwen2.5-7B-Instruct + a LoRA "template" adapter
on a Kaggle GPU to parse requests and disambiguate categories. That
genuinely cannot run on GitHub Actions as-is — Actions runners have no
GPU, and even CPU inference of a 7B model would need the ~14GB merged
fp16 weights (or a multi-GB quantized GGUF) either re-downloaded every
run or committed to the repo, which blows past GitHub's 100MB-per-file
limit the same way the business-data parquet always did.

The fix isn't a workaround, it's removing a dependency that was never
load-bearing for THIS job: per this worker's own long-standing design
note (kept below), parsing an SBA request and matching a category word
to the real taxonomy are both deterministic, verifiable operations with
one right answer that already lives in the real category list — not a
question of "sound more like an analyst," which is the one thing
fine-tuning would actually buy. So this script now does that matching
with plain regex + fuzzy string scoring against the REAL, LIVE category
list (same list, same render-service /api/categories call as before) —
no model load, no GPU, no Kaggle secrets, runs in seconds on a bare
Actions runner. If real use ever surfaces requests this genuinely can't
parse reliably (ambiguous phrasing a human would also struggle with, not
just an unhandled pattern), that's the moment to reconsider a model in
the loop for just that step — not before.

Triggered via .github/workflows/research.yml (workflow_dispatch, a form
with a free-text "request" input) or directly: `python company_scrapper_ai.py
"find dental clinics with no website within 3 miles of 123 Main St,
Columbus, OH"`. Produces files under --out-dir, which the workflow uploads
as a downloadable run artifact.
"""

import os
import re
import sys
import json
import difflib
import argparse

import requests

import service_sba_trade_area_snapshot as sba

# The live read API (render-service) — same URL the GitHub Pages dashboard
# and service_sba_trade_area_snapshot.py's own default call. Sharded behind
# the scenes (render-service/lib/db.js fans out across every configured
# DATABASE_URL_N and merges) — none of which this script needs to know
# about, it's one URL from here.
RENDER_API_BASE = os.environ.get("RENDER_API_BASE", "https://aicompany-tmg7.onrender.com")


# ---------------------------------------------------------------------
# Real category taxonomy — pulled LIVE from the actual live database via
# render-service's API, never invented and never a stale local snapshot.
# ---------------------------------------------------------------------
def load_real_categories(api_base=RENDER_API_BASE):
    r = requests.get(f"{api_base}/api/categories", timeout=30)
    r.raise_for_status()
    return sorted(r.json()["categories"])


def _singularize_word(w):
    """Crude but effective for this domain: handles the two common
    English plural shapes separately rather than one blind rule — a
    trailing 'ies' singularizes to 'y' ('pharmacies'->'pharmacy',
    'bakeries'->'bakery'), which a plain trailing-'s' strip would mangle
    into 'pharmacie'/'bakerie' and miss the match entirely; everything
    else over 3 letters just drops a trailing 's' ('shops'->'shop',
    'clinics'->'clinic')."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def _singularize_words(s):
    return {_singularize_word(w) for w in s.split()}


def _category_score(term: str, norm_cat: str) -> float:
    """Word-identity-first scoring, NOT raw character similarity — two
    unrelated category phrases of similar length and vocabulary (e.g.
    "coffee shops" vs. "auto repair shop") can score deceptively high on
    difflib's character-level ratio alone just from sharing common
    letters/spacing, which previously let a wrong category sneak into
    the matched list and silently broaden a client's trade-area results
    to include a competitor category they never asked about. Exact or
    singularized-word-set matches score highest; character ratio is kept
    only as a heavily-discounted fallback for genuine near-miss spelling
    ("vetrinary" -> "veterinary"), never enough on its own to beat a real
    word-level mismatch."""
    if term == norm_cat:
        return 1.0
    term_sing, cat_sing = _singularize_words(term), _singularize_words(norm_cat)
    if term_sing == cat_sing:
        return 1.0
    if term in norm_cat or norm_cat in term:
        return 0.9
    union = term_sing | cat_sing
    jaccard = len(term_sing & cat_sing) / len(union) if union else 0.0
    char_ratio = difflib.SequenceMatcher(None, term, norm_cat).ratio()
    return max(jaccard * 0.85, char_ratio * 0.5)


def match_categories(plain_english_term, real_categories, top_k=3, min_score=0.45):
    """Maps what a user actually says ('dentist', 'coffee shop') to the
    REAL category strings in the database — purely deterministic (no
    model), scored against both the real string's space-normalized form
    (Overture's real taxonomy values are typically snake_case,
    "dental_clinic", "coffee_shop", while a user types them with spaces)
    via _category_score. Only returns matches within a real margin of the
    single best score — a weak secondary match doesn't get smuggled into
    the list just because it cleared the floor, since every returned
    category is OR'd together in the live query and a wrong one silently
    broadens what the client sees. Returns real category strings only,
    ranked best-first, never a fabricated one."""
    term = (plain_english_term or "").strip().lower()
    if not term:
        return []

    scored = [(_category_score(term, cat.replace("_", " ").lower()), cat) for cat in real_categories]
    scored = [(s, c) for s, c in scored if s >= min_score]
    if not scored:
        return []
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0]
    return [c for s, c in scored[:top_k] if s >= best - 0.2]


# ---------------------------------------------------------------------
# Deterministic request parsing — regex + known-vocabulary matching
# instead of a generative model. Every extracted field is either a plain
# pattern match (radius, "no website") or validated against the real,
# live category list (see _split_category_and_address) — nothing here is
# a free-form guess at what the user meant.
# ---------------------------------------------------------------------
_RADIUS_RE = re.compile(
    r"(?:within\s+)?(\d+(?:\.\d+)?)\s*[- ]?mi(?:les?)?\b(?:\s+radius)?(?:\s+of\b)?",
    re.IGNORECASE,
)
_NO_WEBSITE_RE = re.compile(
    r"(?:with\s+|that\s+(?:have|has)\s+)?(?:no|without|missing)\s+(?:a\s+|an?y\s+)?website",
    re.IGNORECASE,
)
_LEAD_FILLER = (
    "please find", "please show", "please search for", "please get me",
    "find", "search for", "get me", "show me", "look for", "i need", "i want",
)
_CONNECTORS = ("near", "around", "in", "of", "with", "for", "at")


def _strip_words(text, words_tuple):
    """Strips any of words_tuple repeatedly from the START of text — used
    for both filler ('find', 'show me') and connectors ('near', 'of')
    left dangling at the seam where the category phrase was cut out.
    A plain string.replace/regex-with-leading-space approach misses the
    case where the connector IS the first word (no leading space to
    match), which is exactly the common case here ("near 123 Main St..."
    right after "hair salons" is removed) — this strips by word, not by
    assuming surrounding whitespace."""
    t = text.strip()
    changed = True
    while changed:
        changed = False
        low = t.lower()
        for w in words_tuple:
            if low.startswith(w + " ") or low == w:
                t = t[len(w):].strip()
                changed = True
                break
    return t


def _strip_leading_filler(text):
    return _strip_words(text, _LEAD_FILLER)


def _split_category_and_address(text, real_categories):
    """Scans every 1-4 word window of the remaining text (after radius/
    no-website phrases and leading filler words are stripped) for the one
    that best fuzzy-matches a REAL category — the same real, live list
    match_categories() validates against, so this never invents a
    category to split on. Whatever's outside that window becomes the
    address candidate, after trimming connector words/punctuation from
    the join point. Returns (category_phrase, address) — category_phrase
    is the user's own words (still just a plain-English phrase at this
    point; match_categories() does the real validated mapping
    afterward), address may be empty if nothing looks like a category."""
    words = text.strip(" ,.").split()
    if not words:
        return "", ""

    norm_cats = [c.replace("_", " ").lower() for c in real_categories]
    best_score, best_span = 0.0, None
    for size in range(1, min(5, len(words) + 1)):
        for i in range(0, len(words) - size + 1):
            window = " ".join(words[i:i + size]).strip(" ,.").lower()
            if not window:
                continue
            best_for_window = max((_category_score(window, nc) for nc in norm_cats), default=0.0)
            # Slight bonus for longer windows at an equal score — prefer
            # "coffee shop" over just "coffee" when both score the same.
            score = best_for_window + size * 0.001
            if score > best_score:
                best_score, best_span = score, (i, i + size)

    if best_span is None or best_score < 0.45:
        return "", " ".join(words).strip(" ,.")

    i0, i1 = best_span
    category_phrase = " ".join(words[i0:i1]).strip(" ,.")
    remainder_words = words[:i0] + words[i1:]
    remainder = _strip_words(" ".join(remainder_words), _CONNECTORS)
    remainder = re.sub(r"\s{2,}", " ", remainder).strip(" ,.")
    return category_phrase, remainder


def parse_request(nl_request: str, real_categories) -> dict:
    """Extracts the structured pieces of an SBA trade-area request from
    free text, deterministically. Every field is either a direct pattern
    match or validated against real, live data downstream — nothing here
    decides the final answer, it only decides what to ask for."""
    text = nl_request.strip()

    radius_miles = None
    m = _RADIUS_RE.search(text)
    if m:
        radius_miles = float(m.group(1))
        text = (text[:m.start()] + " " + text[m.end():]).strip()

    no_website_only = False
    m2 = _NO_WEBSITE_RE.search(text)
    if m2:
        no_website_only = True
        text = (text[:m2.start()] + " " + text[m2.end():]).strip()

    text = _strip_leading_filler(text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.")

    category, address = _split_category_and_address(text, real_categories)
    return {
        "address": address or None,
        "radius_miles": radius_miles,
        "category": category,
        # This whole platform only ever scans the US + the territories SBA
        # lends in (see scanner-service's module docstring) — defaulting
        # here instead of trying to detect a country from free text avoids
        # a real ambiguity (e.g. "CA" could mean California or Canada).
        "country": "US",
        "no_website_only": no_website_only,
    }


def research(nl_request: str, output_formats=None, out_dir="./company_scrapper_out",
             api_base=RENDER_API_BASE) -> dict:
    """The main entry point: plain-English request in, real results out.
    Every field in the returned spec either came from the user's own
    words or was validated against real data (real geocode, real category
    match against the live list, real row counts) — nothing here is
    invented to fill a gap."""
    real_categories = load_real_categories(api_base)
    print(f"Loaded {len(real_categories):,} real distinct category values from the live database.")

    parsed = parse_request(nl_request, real_categories)
    print(f"Parsed request: {parsed}")

    if not parsed.get("address"):
        raise ValueError("Request has no address/location to anchor the trade area — "
                          "ask for one rather than guessing a location.")

    categories = match_categories(parsed.get("category", ""), real_categories)
    if not categories:
        raise ValueError(f"Could not match '{parsed.get('category')}' to any real category in the "
                          f"database — ask to clarify rather than scanning everything.")
    print(f"Matched category '{parsed.get('category')}' -> real categories: {categories}")

    lat, lon = sba.geocode_address(parsed["address"])
    print(f"Geocoded '{parsed['address']}' -> ({lat:.4f}, {lon:.4f})")

    spec = {
        "center_lat": lat, "center_lon": lon,
        "radius_miles": parsed.get("radius_miles") or 3,
        "category": categories,
        "country": parsed.get("country"),
        "no_website_only": bool(parsed.get("no_website_only")),
        "output_formats": output_formats or ["csv", "pdf"],
        "api_base": api_base,
    }
    preview = sba.preview_trade_area(spec)
    print(f"Preview: {preview}")
    files = sba.fulfil(spec, out_dir)
    return {"spec": spec, "preview": preview, "files": files}


def main():
    ap = argparse.ArgumentParser(description="Company Scrapper — plain-English SBA trade-area research")
    ap.add_argument("request", nargs="?", help="Plain-English research request")
    ap.add_argument("--formats", default="csv,pdf", help="Comma-separated output formats (csv,xlsx,pdf)")
    ap.add_argument("--out-dir", default="./company_scrapper_out")
    ap.add_argument("--api-base", default=RENDER_API_BASE)
    args = ap.parse_args()

    nl_request = args.request or os.environ.get("RESEARCH_REQUEST")
    if not nl_request:
        print("No request given — pass it as a CLI argument, or set RESEARCH_REQUEST.")
        sys.exit(1)

    result = research(
        nl_request,
        output_formats=[f.strip() for f in args.formats.split(",") if f.strip()],
        out_dir=args.out_dir,
        api_base=args.api_base,
    )
    print("\n" + json.dumps(
        {"spec": result["spec"], "preview": result["preview"], "files": result["files"]},
        indent=2, default=str,
    ))


if __name__ == "__main__":
    main()
