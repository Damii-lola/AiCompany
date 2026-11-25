"""
Service execution module: SBA Trade-Area Competitor Snapshot
==============================================================
Second "fulfilment_module" in the platform (see service_lead_research_report.py
for the pattern this follows: `fulfil(spec, out_dir) -> list[file_path]`,
nothing fabricated, SpecError raised rather than guessed around).

This is the real locked V1 SBA feasibility pre-screening product:
"competitor count/list by category within a radius-based trade area, plus
a no-website breakdown." Sold to community-bank/credit-union SBA 7(a)/504
lenders as a fast $200-$500 pre-screen before they order a full $4,900
feasibility study from an incumbent like MMCG.

DATA SOURCE (changed Nov 2026): this used to query a local
business_database.parquet file directly with DuckDB. That file doesn't
exist anymore — the real data now lives in the live, sharded TiDB
Postgres setup (database/schema.sql, scanner-service/scan_worker.py,
render-service), which already did the one-time US+territories backfill
(~18.8M real rows across however many shard databases are configured).
This module is now a thin client against render-service's live HTTP API
(the same API the GitHub Pages dashboard calls) instead of re-implementing
radius/category SQL against a file — render-service already does the
real work (geocoding, category matching, radius search, cross-shard
merge-sort by real distance) and exposes it as JSON.

Two ways to call fulfil(), matching the two ways render-service itself
supports a trade-area lookup:

  (a) Loose address + plain-English category (recommended — this is what
      a non-technical operator would actually have on hand): server
      geocodes the address (Nominatim) and fuzzy-matches the category
      term against the live distinct category list.
        spec = {
            "address": "123 Main St, Columbus, OH",
            "category": "dental clinic",
            "radius_miles": 3,
        }

  (b) Exact coordinates + exact real category string(s) (for a caller
      that already has precise data, e.g. a previous geocode result or a
      category pulled straight from /api/categories): no fuzzy matching,
      no geocoding round-trip.
        spec = {
            "center_lat": 39.9612, "center_lon": -82.9988,
            "category": ["dental clinic", "orthodontist"],
            "radius_miles": 3,
        }

Common optional spec fields (either form): country, no_website_only,
exclude_categories (list, applied client-side — render-service's API has
no concept of this), operating_status_open_only (client-side too),
output_formats (default ["csv"]), api_base (override the render-service
URL, default below).
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone

from service_lead_research_report import SpecError, _normalize_phone
from service_business_enrichment import enrich_businesses, enrichment_summary
from service_business_registry import enrich_with_registry, registry_summary
from service_competitive_classifier import classify_businesses, classification_summary

# The live read API (render-service) — same URL the GitHub Pages
# dashboard calls. Overridable per-call via spec["api_base"] for a local
# render-service instance during testing.
DEFAULT_API_BASE = "https://aicompany-tmg7.onrender.com"


def geocode_address(address: str, timeout: int = 15) -> tuple:
    """Turns a free-text address into (lat, lon) using OpenStreetMap's
    Nominatim directly (no API key, best-effort). Only needed if a caller
    wants raw coordinates for some OTHER purpose — fulfil() itself prefers
    to let render-service's /api/research do this server-side (form (a)
    above) rather than geocoding twice. Raises SpecError rather than
    returning a plausible-looking wrong coordinate."""
    headers = {"User-Agent": "sba-trade-area-snapshot/1.0 (internal tool)"}
    params = {"q": address, "format": "json", "limit": 1}
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", params=params,
                          headers=headers, timeout=timeout)
        r.raise_for_status()
        results = r.json()
    except Exception as e:
        raise SpecError(f"Geocoding request failed for '{address}': {e}") from e
    if not results:
        raise SpecError(f"Geocoding found no match for '{address}' — "
                         f"get coordinates another way, don't guess them.")
    return float(results[0]["lat"]), float(results[0]["lon"])


def _require(spec: dict, *fields):
    missing = [f for f in fields if spec.get(f) in (None, "", [])]
    if missing:
        raise SpecError(f"Trade-area spec is missing required field(s): {', '.join(missing)}.")


def _fetch_trade_area(spec: dict) -> dict:
    """Calls the live render-service API and returns its JSON response
    verbatim ({summary, businesses, ...}). Raises SpecError on anything
    that isn't a clean 200 — a network failure, a bad address, or a
    category that matched nothing are all real conditions a caller needs
    to know about, not something to paper over."""
    api_base = spec.get("api_base", DEFAULT_API_BASE).rstrip("/")
    _require(spec, "radius_miles", "category")

    if spec.get("address"):
        if not isinstance(spec["category"], str):
            raise SpecError(
                "When using 'address' (form (a), loose/plain-English lookup), "
                "'category' must be a single plain-English string — render-service's "
                "/api/research fuzzy-matches it server-side. Pass 'center_lat'/'center_lon' "
                "plus an exact category string or list instead (form (b)) if you already "
                "have precise coordinates/category names."
            )
        try:
            r = requests.post(f"{api_base}/api/research", json={
                "address": spec["address"],
                "category": spec["category"],
                "radius_miles": spec["radius_miles"],
                "no_website_only": bool(spec.get("no_website_only", False)),
                "country": spec.get("country"),
            }, timeout=30)
        except Exception as e:
            raise SpecError(f"Could not reach the live business database at {api_base}: {e}") from e
        if r.status_code == 422:
            body = r.json()
            raise SpecError(
                f"{body.get('error', 'Category did not match anything real.')} "
                f"Real category examples: {body.get('real_category_sample', [])[:10]}"
            )
        if not r.ok:
            raise SpecError(f"Live business database returned {r.status_code}: {r.text[:500]}")
        return r.json()

    _require(spec, "center_lat", "center_lon")
    cat = spec["category"]
    cats = [cat] if isinstance(cat, str) else list(cat)
    try:
        r = requests.get(f"{api_base}/api/trade-area", params={
            "lat": spec["center_lat"], "lon": spec["center_lon"],
            "radius_miles": spec["radius_miles"], "category": ",".join(cats),
            "no_website_only": "true" if spec.get("no_website_only") else "false",
            "country": spec.get("country") or "",
        }, timeout=30)
    except Exception as e:
        raise SpecError(f"Could not reach the live business database at {api_base}: {e}") from e
    if not r.ok:
        raise SpecError(f"Live business database returned {r.status_code}: {r.text[:500]}")
    return r.json()


def _apply_client_side_filters(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Filters render-service has no notion of (exclude_categories,
    operating_status_open_only) — applied here, after the fact, against
    the real rows the API returned. Everything downstream (counts,
    no-website breakdown, by-category table, nearest-competitor) is then
    computed from THIS filtered frame, same as before this module talked
    to a live API instead of a local file — stats were always derived
    from the actual result set, never from a separately-trusted count."""
    if spec.get("exclude_categories"):
        ex = spec["exclude_categories"]
        ex = {ex} if isinstance(ex, str) else set(ex)
        df = df[~df["category"].isin(ex)]
    if spec.get("operating_status_open_only"):
        closed = {"closed", "permanently_closed"}
        df = df[df["operating_status"].isna() | ~df["operating_status"].str.lower().isin(closed)]
    return df


def preview_trade_area(spec: dict) -> dict:
    """Real counts, no padding — call this before promising a client
    anything. Hits the live API and reports what's actually there."""
    result = _fetch_trade_area(spec)
    df = pd.DataFrame(result["businesses"])
    if not df.empty:
        df = _apply_client_side_filters(df, spec)
    return {
        "competitor_count": len(df),
        "no_website_count": int((df["website"].isna() | (df["website"].str.strip() == "")).sum()) if not df.empty else 0,
        "radius_miles": spec["radius_miles"],
        "category": spec["category"],
        "matched_categories": result.get("matched_categories"),
        "center": result.get("center"),
    }


def fulfil(spec: dict, out_dir: str) -> list:
    """Produces the trade-area competitor snapshot deliverable(s) from
    the LIVE sharded database. Returns the list of file paths actually
    written. Raises SpecError on zero matches rather than delivering an
    empty file that looks like a bug."""
    os.makedirs(out_dir, exist_ok=True)

    result = _fetch_trade_area(spec)
    df = pd.DataFrame(result["businesses"])
    if not df.empty:
        df = _apply_client_side_filters(df, spec)

    if df.empty:
        raise SpecError(
            f"Zero competitors found for category={spec['category']!r} within "
            f"{spec['radius_miles']} miles. Either the trade area genuinely has none "
            f"(a real, reportable finding — widen the radius or confirm with the client "
            f"before treating this as an error) or the category/address didn't resolve "
            f"to what you expected — matched_categories/center in the API response "
            f"(not available here since there were zero rows) would normally confirm that."
        )

    df["phone"] = df["phone"].apply(lambda p: _normalize_phone(p, spec.get("country")))
    df["distance_miles"] = df["distance_miles"].round(2)
    df = df.sort_values("distance_miles").reset_index(drop=True)

    # On-demand enrichment — real website checks (schema.org hours/rating/
    # price/social, reachability, mobile-friendliness), run ONLY for the
    # businesses actually landing in this one report, never against the
    # bulk 18.8M-row index. See service_business_enrichment.py's module
    # docstring for the full cost reasoning. Disable per-call with
    # spec["enrich"] = False (e.g. for a fast preview-only run).
    enrichment_stats = None
    registry_stats = None
    classification_stats = None
    if spec.get("enrich", True):
        records = enrich_businesses(df.to_dict("records"))
        enrich_df = pd.json_normalize([r["enrichment"] for r in records], sep="_")
        enrich_df.columns = [f"enrich_{c}" for c in enrich_df.columns]
        df = pd.concat([df.reset_index(drop=True), enrich_df.reset_index(drop=True)], axis=1)
        enrichment_stats = enrichment_summary(records)

        # Second, independent enrichment tier: real state Secretary-of-
        # State registry cross-reference (legitimacy/status). Runs off
        # the SAME `records` list (already has distance order + names),
        # not `df`, since it only needs name/region, not the website
        # fields the first pass just added.
        registry_records = enrich_with_registry(records)
        registry_df = pd.json_normalize([r["registry"] for r in registry_records], sep="_")
        registry_df.columns = [f"registry_{c}" for c in registry_df.columns]
        df = pd.concat([df.reset_index(drop=True), registry_df.reset_index(drop=True)], axis=1)
        registry_stats = registry_summary(registry_records)

        # Third tier: competitive classification (direct/indirect/related/
        # irrelevant) against the category the customer actually searched
        # — deterministic, grounded in Overture's own real taxonomy
        # hierarchy when the scanned data has it, honestly falls back to
        # leaf-category comparison when it doesn't. See
        # service_competitive_classifier.py's module docstring for why
        # this is a rule-based tier and not the trained worker-AI
        # adapter the project reserves true subjective judgment calls for.
        classified_records = classify_businesses(spec.get("category", ""), registry_records)
        classification_df = pd.json_normalize(
            [r["competitive_classification"] for r in classified_records], sep="_")
        classification_df.columns = [f"classification_{c}" for c in classification_df.columns]
        df = pd.concat([df.reset_index(drop=True), classification_df.reset_index(drop=True)], axis=1)
        classification_stats = classification_summary(classified_records)

    no_website = df[df["website"].isna() | (df["website"].astype(str).str.strip() == "")]
    by_category = df["category"].value_counts()

    files = []
    formats = spec.get("output_formats", ["csv"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    export_cols = ["name", "category", "category_extra", "address", "city", "region",
                   "postcode", "country", "phone", "website", "email", "social", "brand",
                   "operating_status", "lat", "lon", "distance_miles",
                   "enrich_website_status", "enrich_hours_text", "enrich_price_range",
                   "enrich_rating_value", "enrich_review_count", "enrich_mobile_friendly",
                   "enrich_social_links", "enrich_phone_confirmed",
                   "registry_matched", "registry_status_text", "registry_registry_name",
                   "classification_classification", "classification_basis"]
    export_cols = [c for c in export_cols if c in df.columns]

    if "csv" in formats:
        path = os.path.join(out_dir, f"trade_area_competitors_{stamp}.csv")
        df[export_cols].to_csv(path, index=False)
        files.append(path)

    if "xlsx" in formats:
        path = os.path.join(out_dir, f"trade_area_competitors_{stamp}.xlsx")
        df[export_cols].to_excel(path, index=False)
        files.append(path)

    if "pdf" in formats:
        path = os.path.join(out_dir, f"trade_area_snapshot_{stamp}.pdf")
        _write_pdf_snapshot(df, no_website, by_category, spec, result, path,
                            enrichment_stats, registry_stats, classification_stats)
        files.append(path)

    manifest = {
        "spec": spec,
        "matched_categories": result.get("matched_categories"),
        "center": result.get("center"),
        "competitor_count": len(df),
        "no_website_count": len(no_website),
        "no_website_pct": round(100 * len(no_website) / len(df), 1),
        "nearest_competitor_miles": float(df["distance_miles"].min()),
        "by_category": by_category.to_dict(),
        # Real on-demand enrichment coverage — how many of the rows above
        # actually got a live website check this run, not guessed figures.
        "enrichment": enrichment_stats,
        "registry_check": registry_stats,
        "competitive_classification": classification_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    files.append(manifest_path)

    return files


def _write_pdf_snapshot(df: pd.DataFrame, no_website: pd.DataFrame, by_category: pd.Series,
                         spec: dict, result: dict, path: str, enrichment_stats: dict = None,
                         registry_stats: dict = None, classification_stats: dict = None):
    """One-page trade-area snapshot: this is the document an SBA loan
    applicant or their lender actually reads, not the raw CSV."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    center = result.get("center") or {"lat": spec.get("center_lat"), "lon": spec.get("center_lon")}
    category_label = spec.get("address") and result.get("matched_categories") or spec["category"]

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = [
        Paragraph("SBA Trade-Area Competitor Snapshot", styles["Title"]),
        Spacer(1, 0.1 * inch),
        Paragraph("Pre-screen reference only — not a substitute for a full feasibility study.",
                  styles["Italic"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"Location: {spec.get('address', '')} ({center.get('lat'):.4f}, {center.get('lon'):.4f})  |  "
                  f"Radius: {spec['radius_miles']} miles  |  Category: {category_label}",
                  styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>{len(df)}</b> competitors found in trade area", styles["Heading2"]),
        Paragraph(f"<b>{len(no_website)}</b> ({round(100*len(no_website)/len(df),1)}%) have no listed website",
                  styles["Normal"]),
        Paragraph(f"Nearest competitor: {df['distance_miles'].min():.2f} miles away", styles["Normal"]),
        Spacer(1, 0.2 * inch),
    ]

    if enrichment_stats and enrichment_stats.get("websites_checked"):
        es = enrichment_stats
        story.append(Paragraph(
            f"Live website check: {es['websites_reachable']}/{es['websites_checked']} competitor "
            f"websites verified reachable this run"
            + (f", avg rating {es['avg_rating_where_available']}/5 across "
               f"{es['rating_sample_size']} with a published rating" if es.get("avg_rating_where_available") else "")
            + f", {es['listings_with_hours_published']} publish real hours online.",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.2 * inch))

    if classification_stats and (classification_stats["direct"] or classification_stats["indirect"]
                                  or classification_stats["related"]):
        cs = classification_stats
        story.append(Paragraph(
            f"Competitive classification: {cs['direct']} direct competitor(s), "
            f"{cs['indirect']} indirect, {cs['related']} related, {cs['irrelevant']} irrelevant "
            f"(deterministic, based on Overture's own category taxonomy where available — "
            f"{cs['classified_from_real_hierarchy_data']} of these used real hierarchy data, "
            f"{cs['classified_from_leaf_category_fallback']} used the leaf-category fallback).",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.2 * inch))

    if registry_stats and registry_stats.get("registry_checks_attempted"):
        rs = registry_stats
        story.append(Paragraph(
            f"State registry check: {rs['states_with_registry_coverage']}/{rs['registry_checks_attempted']} "
            f"competitors are in a state with registry coverage, {rs['confirmed_registered_entities']} "
            f"confirmed as registered legal entities"
            + (f" (statuses seen: {', '.join(rs['status_values_found'])})" if rs["status_values_found"] else "") + ".",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.2 * inch))

    story.append(Spacer(1, 0.1 * inch))

    rows = [["Category", "Count"]] + [[c, str(n)] for c, n in by_category.head(15).items()]
    t = Table(rows, colWidths=[3 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    doc.build(story)


def run_smoke_test(api_base: str = DEFAULT_API_BASE):
    """Call this yourself to sanity-check against the real live API —
    not auto-run on import. Exercises BOTH call forms."""
    spec_a = {
        "address": "Columbus, OH",
        "category": "dental clinic",
        "radius_miles": 5,
        "country": "US",
        "output_formats": ["csv", "pdf"],
        "api_base": api_base,
    }
    print("Form (a) — address + loose category:")
    preview = preview_trade_area(spec_a)
    print("  Preview:", preview)
    if preview["competitor_count"] > 0:
        files = fulfil(spec_a, "./test_trade_area_a")
        print("  Produced:", files)
    else:
        print("  Zero competitors — real finding, not an error.")


if __name__ == "__main__":
    import sys
    run_smoke_test(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_API_BASE)
