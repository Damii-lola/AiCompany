"""
State business-registry cross-reference (legitimacy/status verification)
==========================================================================
Second on-demand enrichment tier (see service_business_enrichment.py for
the first — website/schema.org checks). This one answers the SBA spec's
"status/legitimacy" category for real: is this business actually a
registered legal entity in its state, and what does the state itself say
its status is (active / dissolved / etc.)?

DATA SOURCE: state Secretary-of-State business-registration datasets
published as open data on each state's own Socrata portal — these are
real government filings, not scraped HTML, free, keyless for the request
volume one report needs (an optional free Socrata app token raises the
per-IP rate limit further; see SOCRATA_APP_TOKEN below, never required).
Five states are wired in, matching the real coverage found during this
project's research (a public gist cataloguing which state portals carry
business-entity data, cross-checked here against each dataset's own
catalog.data.gov listing):
  NY  data.ny.gov        n9v6-gdp6   (Active Corporations, confirmed via
                                       catalog.data.gov, ~4.22M rows)
  CO  data.colorado.gov  4ykn-tg5h   (confirmed — fields verified: entityname,
                                       entitystatus, principalcity, etc.)
  OR  data.oregon.gov    tckn-sxa6   (Active Businesses - ALL, confirmed via
                                       catalog.data.gov)
  CT  data.ct.gov        n7gp-d28j   (Business Registry - Business Master,
                                       confirmed via catalog.data.gov)
  PA  data.pa.gov        xvd7-5r2c   (NOT independently domain-confirmed —
                                       this sandbox can only reach data.gov-
                                       catalog pages, not data.pa.gov itself,
                                       to verify. Code path is identical to
                                       the confirmed four, but LOOKUP_ERRORS
                                       will show real failures here until
                                       someone with real network access
                                       confirms it once. Never silently
                                       trusted over a confirmed source.)

Most of these states' EXACT column names were never independently
confirmed either (only Colorado's were, from the research source) — so
rather than hardcode guessed field names for NY/OR/CT/PA and risk silently
reading the wrong column, this module uses Socrata's own `$q` full-text
search (searches every column) to find candidate rows, then HEURISTICALLY
locates whichever returned field name contains "name"/"status"/"city"
(case-insensitive substring) to read from. This is honest about what
isn't verified: a real search against real government data, read generically
rather than guessed into specific field names that could be wrong.

Every match is additionally confirmed by comparing the returned name
field against the business's actual name (normalized, substring-tolerant)
before being accepted — a loose full-text search can and does return
unrelated businesses, so nothing is reported as "found" on a city/state
match alone.

Never raises on a single business or a single state's failure — one
state portal being down must not break enrichment for the other four, or
for website-based enrichment running alongside it.
"""

import os
import re
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

USER_AGENT = "company-scrapper-registry-check/1.0 (+internal SBA trade-area report tool)"
REQUEST_TIMEOUT = float(os.environ.get("REGISTRY_REQUEST_TIMEOUT", "8"))
MAX_WORKERS = int(os.environ.get("REGISTRY_MAX_WORKERS", "6"))
BUDGET_SECONDS = float(os.environ.get("REGISTRY_BUDGET_SECONDS", "90"))
MAX_CANDIDATES = int(os.environ.get("REGISTRY_MAX_CANDIDATES", "150"))
# Optional — raises Socrata's per-app-token throughput limit; completely
# unnecessary at the scale one report needs (a few hundred requests),
# Socrata's unauthenticated limit already covers that comfortably.
SOCRATA_APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN")

STATE_SOCRATA_SOURCES = {
    "NY": {"domain": "data.ny.gov", "resource_id": "n9v6-gdp6", "domain_confirmed": True},
    "CO": {"domain": "data.colorado.gov", "resource_id": "4ykn-tg5h", "domain_confirmed": True,
           "name_field": "entityname", "status_field": "entitystatus", "city_field": "principalcity"},
    "OR": {"domain": "data.oregon.gov", "resource_id": "tckn-sxa6", "domain_confirmed": True},
    "CT": {"domain": "data.ct.gov", "resource_id": "n7gp-d28j", "domain_confirmed": True},
    "PA": {"domain": "data.pa.gov", "resource_id": "xvd7-5r2c", "domain_confirmed": False},
}

_NAME_KEY_RE = re.compile(r"name", re.I)
_STATUS_KEY_RE = re.compile(r"status", re.I)
_CITY_KEY_RE = re.compile(r"city", re.I)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_name(s: str) -> str:
    """Business-name normalization for fuzzy comparison — strips legal
    suffixes/punctuation/case so 'Joe's Diner, LLC' and 'JOES DINER LLC'
    compare equal without needing exact-string matching against a loose
    full-text search result."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(llc|inc|incorporated|corp|corporation|co|ltd|lp|llp|pllc|pc)\b\.?", "", s)
    return _NON_ALNUM_RE.sub("", s).strip()


def _pick_field(row: dict, pattern) -> str:
    for k, v in row.items():
        if pattern.search(k) and isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _name_matches(business_name: str, candidate_name: str) -> bool:
    a, b = _normalize_name(business_name), _normalize_name(candidate_name)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _lookup_one_state(business: dict, state_code: str) -> dict:
    src = STATE_SOCRATA_SOURCES[state_code]
    name = (business.get("name") or "").strip()
    base = dict(
        covered=True, matched=False, registry_name=None, status_text=None,
        city=None, domain=src["domain"], domain_confirmed=src["domain_confirmed"],
        checked_at=datetime.now(timezone.utc).isoformat(), error=None,
    )
    if not name:
        base["error"] = "business has no name on file to search for"
        return base

    url = f"{src.get('scheme', 'https')}://{src['domain']}/resource/{src['resource_id']}.json"
    params = {"$q": name, "$limit": 5}
    headers = {"User-Agent": USER_AGENT}
    if SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = SOCRATA_APP_TOKEN

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        base["error"] = f"request failed: {e}"
        return base
    if resp.status_code >= 400:
        base["error"] = f"HTTP {resp.status_code}"
        return base
    try:
        rows = resp.json()
    except Exception as e:
        base["error"] = f"non-JSON response: {e}"
        return base
    if not isinstance(rows, list):
        base["error"] = "unexpected response shape (not a list of rows)"
        return base

    name_field = src.get("name_field")
    status_field = src.get("status_field")
    city_field = src.get("city_field")

    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_name = (row.get(name_field) if name_field else None) or _pick_field(row, _NAME_KEY_RE)
        if not _name_matches(name, candidate_name):
            continue
        base["matched"] = True
        base["registry_name"] = candidate_name
        base["status_text"] = (row.get(status_field) if status_field else None) or _pick_field(row, _STATUS_KEY_RE) or None
        base["city"] = (row.get(city_field) if city_field else None) or _pick_field(row, _CITY_KEY_RE) or None
        break

    return base


def _lookup_one_business(business: dict) -> dict:
    """Only queries the ONE state matching the business's own region
    field (never all 5 per business) — a dental clinic in Ohio has no
    reason to be checked against New York's registry. Returns
    {"covered": False, ...} cleanly for any state outside the 5 wired in,
    rather than silently skipping with no explanation."""
    region = (business.get("region") or "").strip().upper()
    if region not in STATE_SOCRATA_SOURCES:
        return {
            "covered": False, "matched": False, "registry_name": None,
            "status_text": None, "city": None, "domain": None,
            "domain_confirmed": None, "checked_at": None,
            "error": f"no state registry wired in for region={region!r} yet "
                     f"(covered: {', '.join(sorted(STATE_SOCRATA_SOURCES))})",
        }
    return _lookup_one_state(business, region)


def enrich_with_registry(businesses: list, max_candidates: int = MAX_CANDIDATES,
                          budget_seconds: float = BUDGET_SECONDS,
                          max_workers: int = MAX_WORKERS) -> list:
    """Same contract shape as service_business_enrichment.enrich_businesses:
    returns the SAME list with a `registry` key added to each dict,
    bounded by a candidate cap (nearest-first, since input is already
    distance-sorted) and a wall-clock budget so a big trade area can't
    hang a report."""
    not_checked = {
        "covered": None, "matched": False, "registry_name": None,
        "status_text": None, "city": None, "domain": None,
        "domain_confirmed": None, "checked_at": None,
        "error": "beyond this report's per-run registry-check cap",
    }

    to_check = businesses[:max_candidates]
    started = time.monotonic()
    results_by_id = {}

    if to_check:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_lookup_one_business, b): id(b) for b in to_check}
            for fut in as_completed(futures):
                remaining = budget_seconds - (time.monotonic() - started)
                try:
                    results_by_id[futures[fut]] = fut.result(timeout=max(remaining, 0.1))
                except Exception as e:
                    results_by_id[futures[fut]] = {**not_checked, "error": f"registry worker error: {e}"}
                if time.monotonic() - started > budget_seconds:
                    break

    for b in businesses:
        b["registry"] = results_by_id.get(id(b), dict(not_checked))
    return businesses


def registry_summary(businesses: list) -> dict:
    checked = [b["registry"] for b in businesses if b.get("registry", {}).get("covered") is not None]
    covered = [r for r in checked if r["covered"]]
    matched = [r for r in covered if r["matched"]]
    return {
        "registry_checks_attempted": len(checked),
        "states_with_registry_coverage": len(covered),
        "confirmed_registered_entities": len(matched),
        "status_values_found": sorted({r["status_text"] for r in matched if r["status_text"]}),
    }
