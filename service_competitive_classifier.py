"""
Competitive classification (direct / indirect / related / irrelevant)
========================================================================
The SBA spec's "competitive classification" category is explicitly a
judgment call — the standing project rule is that any real judgment call
is reserved for the platform's own trained worker-AI adapter, never a
third-party LLM, and never faked as a deterministic certainty it isn't.

This module is the DETERMINISTIC, RULE-GROUNDED version of that judgment
— not a replacement for a future trained adapter, but a real, honest,
immediately-usable classifier that ships today, built on real structural
data Overture itself publishes rather than on guessed category pairs.

THE REAL GROUNDING: Overture's current Places schema (v2.0.0, confirmed
directly against docs.overturemaps.org/schema/reference/places/types/
taxonomy/ and the OvertureMaps/schema GitHub repo's place.yaml) gives
every place a `taxonomy.hierarchy` — the FULL general-to-specific category
path, e.g. ["food_and_drink", "restaurant", "casual_eatery",
"gas_station_sushi"] — straight from Overture's own curated taxonomy, not
an external file this project maintains or guesses at. scan_worker.py now
extracts this (as the `category_hierarchy` JSON-array-string column) and
`basic_category` (Overture's own single "cognitively natural" grouping
label) for every newly-scanned business.

Given two businesses' hierarchy paths, classification is just how much of
that path they share, read from the END (specific) backward:
  - identical leaf (full path equal)                       -> direct
  - share the immediate parent (all but the last element)  -> direct
  - share a grandparent-or-nearer ancestor beyond the
    generic top-level category                             -> indirect
  - share only the generic top-level category (e.g. both
    "food_and_drink" but nothing more specific)             -> related
  - no shared ancestor at all                               -> irrelevant

HONESTY ABOUT COVERAGE: category_hierarchy is NULL for every business
scanned before this column existed (the 18.8M-row backfill already done)
and for every OSM-sourced row (OSM tags carry no equivalent taxonomy
concept). This module never guesses a hierarchy that wasn't actually
scanned. When hierarchy data is missing for either business, it falls
back to comparing the flat leaf `category`/`category_extra` strings
directly (exact match only) — a real but less precise signal — and marks
`basis: "leaf_category_fallback"` so a report reader (and this module's
own aggregate stats) can see which classifications rest on the richer
real data versus the fallback.
"""

import json
from functools import lru_cache

# The generic, top-of-tree categories in Overture's taxonomy are too broad
# to mean "related" on their own (practically every retail business is
# under "retail", every eatery under "food_and_drink") — sharing ONLY one
# of these, with nothing more specific in common, is not a meaningful
# competitive signal and is classified irrelevant, not related. This list
# reflects Overture's documented top-level taxonomy categories (the ~20
# or so L0 buckets named across Overture's taxonomy guide and schema
# docs/examples, e.g. "food_and_drink", "retail", "professional_services",
# "health_and_medical", "home_services", etc.) — not exhaustive by
# necessity (Overture's L0 list can grow), so this is a known-list check,
# not a position-in-array check, and anything not recognized here falls
# through to being treated as meaningful (safer default: under-merge
# toward "related" rather than silently over-merge toward "irrelevant").
_GENERIC_TOP_LEVEL = {
    "food_and_drink", "retail", "professional_services", "health_and_medical",
    "accommodation", "arts_and_entertainment", "education", "financial_service",
    "home_services", "personal_care_and_services", "public_service_and_government",
    "religious_organization", "sports_and_recreation", "transportation",
    "automotive", "real_estate", "pet_services", "travel",
}


def _parse_hierarchy(raw) -> list:
    if not raw or not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, str) and x]


def _normalize_leaf(s) -> str:
    return s.strip().lower() if isinstance(s, str) else ""


def classify_pair(target: dict, candidate: dict) -> dict:
    """target: the business/category the report is built around (what the
    customer searched for). candidate: one competitor in the trade area.
    Both accept either a full business dict (category/category_extra/
    basic_category/category_hierarchy keys) or a bare category string
    (treated as the leaf category with no hierarchy data)."""
    if isinstance(target, str):
        target = {"category": target}
    if isinstance(candidate, str):
        candidate = {"category": candidate}

    t_hier = _parse_hierarchy(target.get("category_hierarchy"))
    c_hier = _parse_hierarchy(candidate.get("category_hierarchy"))

    if t_hier and c_hier:
        if t_hier == c_hier:
            return {"classification": "direct", "basis": "hierarchy", "shared_depth": len(t_hier)}
        if t_hier[:-1] == c_hier[:-1] and len(t_hier) > 1 and len(c_hier) > 1:
            return {"classification": "direct", "basis": "hierarchy", "shared_depth": len(t_hier) - 1}
        shared = 0
        for a, b in zip(t_hier, c_hier):
            if a != b:
                break
            shared += 1
        if shared == 0:
            return {"classification": "irrelevant", "basis": "hierarchy", "shared_depth": 0}
        if shared == 1 and t_hier[0] in _GENERIC_TOP_LEVEL:
            return {"classification": "related", "basis": "hierarchy", "shared_depth": 1}
        return {"classification": "indirect", "basis": "hierarchy", "shared_depth": shared}

    # Fallback: no (or only one-sided) hierarchy data — compare flat leaf
    # category strings directly. Less precise, honestly labeled as such.
    t_leaf = _normalize_leaf(target.get("category"))
    c_leaf = _normalize_leaf(candidate.get("category"))
    t_extra = _normalize_leaf(target.get("category_extra"))
    c_extra = _normalize_leaf(candidate.get("category_extra"))

    if not t_leaf or not c_leaf:
        return {"classification": None, "basis": "insufficient_data", "shared_depth": None}

    if t_leaf == c_leaf:
        return {"classification": "direct", "basis": "leaf_category_fallback", "shared_depth": None}
    # A candidate's alternate category matching the target's primary (or
    # vice versa) is a real signal too — Overture's alternates are
    # genuinely-applicable secondary categories for that same business,
    # not a weaker guess.
    if (t_extra and t_extra == c_leaf) or (c_extra and c_extra == t_leaf) or (t_extra and t_extra == c_extra and t_extra):
        return {"classification": "indirect", "basis": "leaf_category_fallback", "shared_depth": None}
    return {"classification": "irrelevant", "basis": "leaf_category_fallback", "shared_depth": None}


def classify_businesses(target_category: str, businesses: list) -> list:
    """Classifies every business in the list against target_category
    (the category the report's trade-area search was actually run for —
    a string, since that's what the customer searched, not a full
    business record). Adds a `competitive_classification` key to each
    business dict; never raises on a single bad record."""
    target = {"category": target_category}
    # The customer's search term has no hierarchy of its own (it's just
    # typed text), but it IS an Overture leaf category name, and real
    # scanned businesses sharing that same leaf category often DO carry
    # real hierarchy data — borrow the first one found as a stand-in
    # reference path for the whole batch, rather than forcing every
    # comparison down to the weaker leaf-only fallback when better data
    # is sitting right there in the batch.
    target_leaf_norm = _normalize_leaf(target_category)
    for b in businesses:
        hier = _parse_hierarchy(b.get("category_hierarchy"))
        if hier and _normalize_leaf(b.get("category")) == target_leaf_norm:
            target["category_hierarchy"] = b["category_hierarchy"]
            break
    for b in businesses:
        try:
            b["competitive_classification"] = classify_pair(target, b)
        except Exception as e:
            b["competitive_classification"] = {
                "classification": None, "basis": "classifier_error",
                "shared_depth": None, "error": str(e),
            }
    return businesses


def classification_summary(businesses: list) -> dict:
    results = [b.get("competitive_classification", {}) for b in businesses]
    counts = {"direct": 0, "indirect": 0, "related": 0, "irrelevant": 0, "unclassified": 0}
    hierarchy_based = 0
    for r in results:
        c = r.get("classification")
        if c in counts:
            counts[c] += 1
        else:
            counts["unclassified"] += 1
        if r.get("basis") == "hierarchy":
            hierarchy_based += 1
    return {
        **counts,
        "classified_from_real_hierarchy_data": hierarchy_based,
        "classified_from_leaf_category_fallback": len(results) - hierarchy_based - counts["unclassified"],
    }
