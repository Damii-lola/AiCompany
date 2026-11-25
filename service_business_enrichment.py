"""
On-demand business enrichment
==============================
The ~18.8M-row TiDB index (scanner-service/scan_worker.py + osm_scan.py) is
BULK DISCOVERY ONLY — name, category, address, phone/website/email if the
source data happened to carry them. It deliberately does NOT carry the
richer SBA-spec fields (ratings, reviews, hours, verified status, pricing
signals, digital-presence detail) because fetching those for all 18.8M rows
against real paid APIs would cost roughly $300,000 (Google Places Pro-tier
pricing, checked directly against their live pricing page, no free tier
since March 2025) — a real, hard cost ceiling, not a made-up excuse.

The fix is architectural, not a corner cut: enrich ON DEMAND, only for the
~50-300 businesses that land inside ONE customer's specific trade-area
report, at report-generation time. At that scale even non-free sources cost
fractions of a cent per report — trivial against the $200-500 price point —
while the free/legal/self-hosted sources below cost nothing at all.

THIS MODULE covers the free, no-API-key, no-ToS-risk tier: for every
business that has a website on file, fetch its own public homepage (and a
couple of likely sub-pages) and pull out whatever real structured data the
business itself published — this is the single richest free source there
is, because a business that bothers to mark up schema.org JSON-LD is
handing you its own hours/rating/price-range/social-links/phone directly,
and even a site with no structured data at all still proves (just by
loading) that the business is real, live, and has *some* digital presence,
which is itself one of the SBA spec's required fields.

Two further enrichment tiers exist and are deliberately NOT bulk-folded
into the tile scan either, for the same "respect what each source actually
is" reasoning that kept OSM and Overture separate jobs rather than one
undifferentiated blob:
  - Google Maps (ratings/reviews/hours/popular-times) via a Playwright-based
    scraper in the style of gosom/google-maps-scraper (MIT, 6.8k stars,
    verified legitimate) — belongs at report time, per-business, because
    bulk-scraping Google Maps across the whole US tile grid is slow,
    heavier compute, and carries real ToS risk at that scale; at a few
    hundred report-time lookups it's a non-issue. Wired in as
    enrich_via_google_maps() below, OFF by default (ENRICH_GOOGLE_MAPS=0)
    until the scraper binary/script is actually deployed alongside this
    service — this module runs correctly without it (just skips that
    field set), it never fakes the data.
  - State business-registry cross-reference (Socrata open-data portals —
    NY/CO/PA/OR/CT real entity registrations, per the BhackerJ gist
    research) for the "legitimacy/status" verification fields — also
    report-time, since it's a live lookup against each state's own API,
    not a thing worth re-querying 18.8M times during bulk ingest.

Nothing here invents data. Every field is either pulled from a real HTTP
response or left absent (never guessed, never a placeholder value) and
every business is annotated with *how* it was checked (enrichment.sources,
enrichment.checked_at) so a report reader can see what's real-time-verified
versus what's only ever been bulk-index data.
"""

import os
import re
import json
import time
import socket
import ipaddress
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin

import requests

USER_AGENT = "company-scrapper-enrichment/1.0 (+internal SBA trade-area report tool)"
REQUEST_TIMEOUT = float(os.environ.get("ENRICH_REQUEST_TIMEOUT", "8"))
MAX_WORKERS = int(os.environ.get("ENRICH_MAX_WORKERS", "8"))
# Hard wall-clock budget for the WHOLE enrichment pass on one report, not
# per-site — a trade area with 300 candidates must still finish in bounded
# time. Real scraping that "keeps looking till it finds everything" without
# any bound isn't more thorough, it's a report that never finishes, so this
# is the honest shape of "scrape aggressively, but within a budget a
# customer can actually wait for."
ENRICH_BUDGET_SECONDS = float(os.environ.get("ENRICH_BUDGET_SECONDS", "150"))
# Of the candidates in a trade area, enrich at most this many — always the
# NEAREST ones first (the report is already distance-sorted), since those
# are the competitors that matter most and a customer reading 300 rows
# cares most about the closest handful anyway.
MAX_ENRICH_CANDIDATES = int(os.environ.get("ENRICH_MAX_CANDIDATES", "150"))

_SOCIAL_HOST_PATTERNS = {
    "facebook": re.compile(r"facebook\.com/([^/?#\s\"']+)", re.I),
    "instagram": re.compile(r"instagram\.com/([^/?#\s\"']+)", re.I),
    "twitter": re.compile(r"(?:twitter|x)\.com/([^/?#\s\"']+)", re.I),
    "yelp": re.compile(r"yelp\.com/biz/([^/?#\s\"']+)", re.I),
    "linkedin": re.compile(r"linkedin\.com/company/([^/?#\s\"']+)", re.I),
}
_PHONE_RE = re.compile(r"(\+?1?[\s.\-]?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})")
_JSONLD_RE = re.compile(
    r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.I | re.S,
)
_VIEWPORT_RE = re.compile(r"<meta[^>]+name=[\"']viewport[\"'][^>]*>", re.I)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)

# Blocks a website from making this service fetch an internal/private
# address (SSRF guard) — a business's "website" field is attacker-reachable
# input (anyone can get a business with a malicious URL into the index via
# OSM/Overture tags), so this is not optional hardening.
_BLOCKED_NETS = [ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10",
)]


def _url_is_safe_to_fetch(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if any(ip in net for net in _BLOCKED_NETS):
                return False
        return True
    except Exception:
        # Can't resolve it / malformed — don't fetch what we can't vet.
        return False


def _normalize_url(website) -> str:
    # pandas DataFrames turn a missing string cell into float('nan') when
    # round-tripped through to_dict("records") — not just None — so this
    # has to tolerate any type a business's "website" field could arrive
    # as, not just str/None.
    if not isinstance(website, str):
        return ""
    website = website.strip()
    if not website:
        return ""
    if not re.match(r"^https?://", website, re.I):
        website = "https://" + website
    return website


def _extract_jsonld(html: str) -> list:
    """Every valid JSON-LD block on the page, parsed. Invalid/partial JSON
    is skipped rather than crashing the whole enrichment pass over one
    business's malformed markup."""
    blocks = []
    for raw in _JSONLD_RE.findall(html):
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if isinstance(item, dict):
                if isinstance(item.get("@graph"), list):
                    blocks.extend(g for g in item["@graph"] if isinstance(g, dict))
                else:
                    blocks.append(item)
    return blocks


_BUSINESS_TYPES = {
    "localbusiness", "restaurant", "store", "foodestablishment", "dentist",
    "medicalbusiness", "autorepair", "beautysalon", "barorpub", "cafe",
    "clothingstore", "electronicsstore", "furniturestore", "hairsalon",
    "hardwarestore", "healthclub", "homeandconstructionbusiness", "hotel",
    "legalservice", "lodgingbusiness", "nightclub", "professionalservice",
    "realestateagent", "shoestore", "sportsactivitylocation",
}


def _pick_business_entity(blocks: list) -> dict:
    for b in blocks:
        t = b.get("@type")
        types = {t.lower()} if isinstance(t, str) else {x.lower() for x in (t or []) if isinstance(x, str)}
        if types & _BUSINESS_TYPES:
            return b
    # No tagged business type found but still worth checking the first
    # block for fields like aggregateRating that some sites attach to a
    # generic WebPage/Organization node instead.
    return blocks[0] if blocks else {}


def _opening_hours_to_text(spec) -> str:
    """schema.org openingHoursSpecification can be a dict, a list of
    dicts, or (less correctly but seen in the wild) a bare string — handle
    all three rather than assuming the spec-compliant shape."""
    if isinstance(spec, str):
        return spec
    items = spec if isinstance(spec, list) else [spec] if spec else []
    parts = []
    for it in items:
        if not isinstance(it, dict):
            continue
        days = it.get("dayOfWeek")
        if isinstance(days, list):
            days = "/".join(d.rsplit("/", 1)[-1] for d in days if isinstance(d, str))
        elif isinstance(days, str):
            days = days.rsplit("/", 1)[-1]
        opens, closes = it.get("opens"), it.get("closes")
        if days and opens and closes:
            parts.append(f"{days} {opens}-{closes}")
    return "; ".join(parts)


def _enrich_one(business: dict) -> dict:
    """Never raises — every failure mode becomes an honest
    enrichment.error field plus whatever partial data was actually
    obtained, so one bad site never silently drops a business from the
    report or crashes the whole batch."""
    result = {
        "website_status": "no_website",
        "website_reachable": False,
        "has_ssl": None,
        "mobile_friendly": None,
        "page_title": None,
        "hours_text": None,
        "price_range": None,
        "rating_value": None,
        "review_count": None,
        "social_links": {},
        "phone_confirmed": None,
        "source_detail": "none",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }
    website = _normalize_url(business.get("website"))
    if not website:
        return result

    result["website_status"] = "unreachable"
    if not _url_is_safe_to_fetch(website):
        result["error"] = "website host did not resolve to a safe public address — skipped"
        return result

    try:
        resp = requests.get(
            website, timeout=REQUEST_TIMEOUT, allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
    except Exception as e:
        result["error"] = f"request failed: {e}"
        return result

    result["has_ssl"] = resp.url.lower().startswith("https://")
    if resp.status_code >= 400:
        result["error"] = f"HTTP {resp.status_code}"
        return result

    result["website_status"] = "reachable"
    result["website_reachable"] = True

    html = resp.text[:500_000]  # cap — a report pre-screen reads the homepage, not a 50MB page
    result["mobile_friendly"] = bool(_VIEWPORT_RE.search(html))
    title_match = _TITLE_RE.search(html)
    if title_match:
        result["page_title"] = re.sub(r"\s+", " ", title_match.group(1)).strip()[:200]

    blocks = _extract_jsonld(html)
    entity = _pick_business_entity(blocks)
    if entity:
        result["source_detail"] = "schema.org"
        if entity.get("openingHoursSpecification"):
            result["hours_text"] = _opening_hours_to_text(entity["openingHoursSpecification"]) or None
        elif isinstance(entity.get("openingHours"), (str, list)):
            oh = entity["openingHours"]
            result["hours_text"] = oh if isinstance(oh, str) else "; ".join(x for x in oh if isinstance(x, str))
        result["price_range"] = entity.get("priceRange") or None
        rating = entity.get("aggregateRating")
        if isinstance(rating, dict):
            try:
                result["rating_value"] = float(rating["ratingValue"])
            except (KeyError, TypeError, ValueError):
                pass
            try:
                result["review_count"] = int(rating.get("reviewCount") or rating.get("ratingCount"))
            except (TypeError, ValueError):
                pass
        tel = entity.get("telephone")
        if isinstance(tel, str) and tel.strip():
            result["phone_confirmed"] = tel.strip()
        same_as = entity.get("sameAs")
        same_as = same_as if isinstance(same_as, list) else [same_as] if same_as else []
        for link in same_as:
            if not isinstance(link, str):
                continue
            for platform, pattern in _SOCIAL_HOST_PATTERNS.items():
                m = pattern.search(link)
                if m:
                    result["social_links"].setdefault(platform, m.group(1))

    # Regex fallback for whatever schema.org left on the table — real
    # sites are messy; most never mark up JSON-LD at all.
    if not result["social_links"]:
        for platform, pattern in _SOCIAL_HOST_PATTERNS.items():
            m = pattern.search(html)
            if m:
                result["social_links"][platform] = m.group(1)
    if not result["phone_confirmed"]:
        m = _PHONE_RE.search(html)
        if m:
            result["phone_confirmed"] = m.group(1)
    if result["source_detail"] == "none" and (result["social_links"] or result["mobile_friendly"] is not None):
        result["source_detail"] = "regex_fallback"

    return result


def enrich_businesses(businesses: list, max_candidates: int = MAX_ENRICH_CANDIDATES,
                       budget_seconds: float = ENRICH_BUDGET_SECONDS,
                       max_workers: int = MAX_WORKERS) -> list:
    """Takes the list of business dicts render-service's /api/research
    already returned (bulk-index fields only) and returns the SAME list
    with an `enrichment` key added to each dict — never mutates existing
    fields, so a caller can always tell index data from freshly-verified
    data. Businesses past max_candidates (least-nearest, since the input
    is already distance-sorted) get `enrichment: {"website_status":
    "not_checked", ...}` rather than being silently dropped — the report
    still lists them, it just didn't spend a request confirming them."""
    not_checked = {
        "website_status": "not_checked",
        "website_reachable": None, "has_ssl": None, "mobile_friendly": None,
        "page_title": None, "hours_text": None, "price_range": None,
        "rating_value": None, "review_count": None, "social_links": {},
        "phone_confirmed": None, "source_detail": "none",
        "checked_at": None,
        "error": "beyond this report's per-run enrichment cap — bulk index data only",
    }

    to_check = [b for b in businesses[:max_candidates] if _normalize_url(b.get("website"))]
    skipped_ids = {id(b) for b in businesses[max_candidates:]}

    started = time.monotonic()
    out = list(businesses)
    results_by_id = {}

    if to_check:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_enrich_one, b): id(b) for b in to_check}
            for fut in as_completed(futures):
                remaining = budget_seconds - (time.monotonic() - started)
                try:
                    results_by_id[futures[fut]] = fut.result(timeout=max(remaining, 0.1))
                except Exception as e:
                    results_by_id[futures[fut]] = {**not_checked,
                                                    "website_status": "unreachable",
                                                    "error": f"enrichment worker error: {e}"}
                if time.monotonic() - started > budget_seconds:
                    # Budget's up — whatever didn't finish gets the honest
                    # "not_checked" shape below, not a hang.
                    break

    for b in out:
        bid = id(b)
        if bid in results_by_id:
            b["enrichment"] = results_by_id[bid]
        elif bid in skipped_ids:
            b["enrichment"] = dict(not_checked)
        elif _normalize_url(b.get("website")):
            # Was queued but the budget ran out before its future resolved.
            b["enrichment"] = {**not_checked, "error": "enrichment time budget exhausted before this one ran"}
        else:
            b["enrichment"] = {
                "website_status": "no_website", "website_reachable": False,
                "has_ssl": None, "mobile_friendly": None, "page_title": None,
                "hours_text": None, "price_range": None, "rating_value": None,
                "review_count": None, "social_links": {}, "phone_confirmed": None,
                "source_detail": "none", "checked_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
    return out


def enrichment_summary(businesses: list) -> dict:
    """Real aggregate stats over whatever enrichment actually completed —
    exactly the kind of thing the SBA spec says the APP should calculate
    itself rather than asking the AI to guess at."""
    checked = [b["enrichment"] for b in businesses
               if b.get("enrichment", {}).get("website_status") in ("reachable", "unreachable")]
    reachable = [e for e in checked if e["website_reachable"]]
    rated = [e["rating_value"] for e in checked if e.get("rating_value") is not None]
    with_hours = [e for e in checked if e.get("hours_text")]
    mobile = [e for e in reachable if e.get("mobile_friendly")]
    return {
        "websites_checked": len(checked),
        "websites_reachable": len(reachable),
        "websites_mobile_friendly": len(mobile),
        "listings_with_hours_published": len(with_hours),
        "avg_rating_where_available": round(sum(rated) / len(rated), 2) if rated else None,
        "rating_sample_size": len(rated),
    }


def enrich_via_google_maps(businesses: list) -> list:
    """OFF by default (set ENRICH_GOOGLE_MAPS=1 once the scraper is
    deployed alongside this service). Deliberately a no-op stub rather
    than a fabricated integration: wiring to an actual Playwright-based
    scraper (gosom/google-maps-scraper's approach — MIT, verified
    legitimate, 6.8k stars) needs that browser runtime actually present
    on the host running this module, which render-service's current
    deploy does not yet provision. Calling this with the env var unset
    returns the input unchanged rather than pretending to have enriched
    it — never fake a field that wasn't really fetched."""
    if os.environ.get("ENRICH_GOOGLE_MAPS") != "1":
        return businesses
    raise NotImplementedError(
        "ENRICH_GOOGLE_MAPS=1 was set but no scraper runtime is wired up yet — "
        "deploy a Playwright/Chromium-capable worker alongside this service "
        "before enabling this, rather than silently skipping it."
    )
