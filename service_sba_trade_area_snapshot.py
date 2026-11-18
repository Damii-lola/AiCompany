"""
Service execution module: SBA Trade-Area Competitor Snapshot
==============================================================
Second "fulfilment_module" in the platform (see service_lead_research_report.py
for the pattern this follows exactly: `fulfil(spec, db_path, out_dir) -> list[file_path]`,
nothing fabricated, SpecError raised rather than guessed around).

This is the real missing piece for the SBA feasibility pre-screening product
(locked V1 definition): "competitor count/list by category within a
radius-based trade area, plus a no-website breakdown." The existing lead
report module only supports exact-match category/country/region/city
filters — it has NO notion of distance. A trade area is a circle around a
proposed business address, not an administrative boundary, so this module
adds real radius (great-circle distance) filtering on top of the same
underlying business_database.parquet (which already has real lon/lat
columns — confirmed from kaggle_business_scanner.py's SELECT clause).

Usage:
    spec = {
        "center_lat": 39.9612,              # required: trade-area center
        "center_lon": -82.9988,             # required
        "radius_miles": 3,                  # required
        "category": "dental_clinic",        # required: competitor category, or list
        "country": "US",                    # optional, same filters as the lead report module
        "output_formats": ["csv", "pdf"],
    }
    files = fulfil(spec, db_path="business_database.parquet", out_dir="./deliverables/sba_job123")

    # or, if you only have a street address and not coordinates yet:
    lat, lon = geocode_address("123 Main St, Columbus, OH")
"""

import os
import json
import time
import duckdb
import pandas as pd
from datetime import datetime, timezone

from service_lead_research_report import SpecError, _normalize_phone, _COUNTRY_CALLING_CODE

EARTH_RADIUS_MILES = 3958.7613


def geocode_address(address: str, timeout: int = 15) -> tuple:
    """Turns a free-text address into (lat, lon) using OpenStreetMap's Nominatim
    (no API key, but rate-limited and best-effort — real geocoding, not a guess).
    Raises SpecError if it can't resolve the address rather than returning a
    plausible-looking wrong coordinate. Requires outbound internet access from
    wherever this runs; if that's not available, pass center_lat/center_lon
    directly in the spec instead of calling this."""
    import requests
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


def _distance_expr(center_lat: float, center_lon: float) -> str:
    """Great-circle (haversine via acos/cos/sin) distance in miles from the
    trade-area center to each row's lon/lat. Real spherical-earth formula,
    not a flat-plane approximation — matters at the radii SBA jobs use
    (1-10 miles) just as much as it would at longer range, since DuckDB
    computes it exactly the same either way for free."""
    return (
        f"({EARTH_RADIUS_MILES} * acos(least(1.0, greatest(-1.0, "
        f"cos(radians({center_lat})) * cos(radians(lat)) * "
        f"cos(radians(lon) - radians({center_lon})) + "
        f"sin(radians({center_lat})) * sin(radians(lat))"
        f"))))"
    )


def _build_trade_area_where(spec: dict) -> tuple:
    """Same filter vocabulary as service_lead_research_report._build_where
    (category/country/region/city/no_website/has_phone/has_email), plus the
    mandatory radius clause this module exists to add. Kept as its own
    function rather than importing _build_where directly because the radius
    clause has to be injected alongside it, not after — DuckDB needs lon/lat
    IS NOT NULL checked before the distance math runs on a row."""
    for required in ("center_lat", "center_lon", "radius_miles", "category"):
        if spec.get(required) in (None, "", []):
            raise SpecError(f"Trade-area spec is missing required field '{required}'.")

    clauses = ["lon IS NOT NULL", "lat IS NOT NULL"]
    params = []

    dist = _distance_expr(spec["center_lat"], spec["center_lon"])
    clauses.append(f"{dist} <= ?")
    params.append(spec["radius_miles"])

    cat = spec["category"]
    cats = [cat] if isinstance(cat, str) else list(cat)
    clauses.append(f"category IN ({','.join(['?'] * len(cats))})")
    params.extend(cats)

    if spec.get("exclude_categories"):
        ex = spec["exclude_categories"]
        ex = [ex] if isinstance(ex, str) else list(ex)
        clauses.append(f"category NOT IN ({','.join(['?'] * len(ex))})")
        params.extend(ex)

    if spec.get("country"):
        clauses.append("country = ?")
        params.append(spec["country"])

    if spec.get("region"):
        clauses.append("region = ?")
        params.append(spec["region"])

    if spec.get("city"):
        clauses.append("city = ?")
        params.append(spec["city"])

    if spec.get("operating_status_open_only"):
        clauses.append("(operating_status IS NULL OR lower(operating_status) NOT IN ('closed', 'permanently_closed'))")

    return " AND ".join(clauses), params, dist


def preview_trade_area(spec: dict, db_path: str) -> dict:
    """Real counts, no padding — call this before promising a client anything.
    Returns total competitor count AND the no-website count, since that's the
    second half of the locked V1 product (competitors + no-website breakdown),
    not just a raw row count."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW biz AS SELECT * FROM read_parquet('{db_path}')")
    where, params, dist = _build_trade_area_where(spec)

    total = con.execute(f"SELECT count(*) FROM biz WHERE {where}", params).fetchone()[0]
    no_site = con.execute(
        f"SELECT count(*) FROM biz WHERE {where} AND (website IS NULL OR trim(website) = '')",
        params,
    ).fetchone()[0]
    con.close()
    return {"competitor_count": total, "no_website_count": no_site,
            "radius_miles": spec["radius_miles"], "category": spec["category"]}


def fulfil(spec: dict, db_path: str, out_dir: str) -> list:
    """Produces the trade-area competitor snapshot deliverable(s). Returns the
    list of file paths actually written. Raises SpecError on zero matches
    rather than delivering an empty file that looks like a bug."""
    os.makedirs(out_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW biz AS SELECT * FROM read_parquet('{db_path}')")
    where, params, dist = _build_trade_area_where(spec)

    cols_to_select = (
        f"name, category, address, city, region, postcode, country, phone, "
        f"website, email, social, brand, operating_status, lat, lon, "
        f"{dist} AS distance_miles"
    )
    df = con.execute(
        f"SELECT {cols_to_select} FROM biz WHERE {where} ORDER BY distance_miles ASC",
        params,
    ).df()
    con.close()

    if df.empty:
        raise SpecError(
            f"Zero competitors found for category={spec['category']!r} within "
            f"{spec['radius_miles']} miles of ({spec['center_lat']}, {spec['center_lon']}). "
            f"Either the trade area genuinely has none (a real, reportable finding — "
            f"widen the radius or confirm with the client before treating this as an error) "
            f"or the category string doesn't match what's in the database."
        )

    df["phone"] = df["phone"].apply(lambda p: _normalize_phone(p, spec.get("country")))
    df["distance_miles"] = df["distance_miles"].round(2)

    no_website = df[df["website"].isna() | (df["website"].str.strip() == "")]
    by_category = df["category"].value_counts()

    files = []
    formats = spec.get("output_formats", ["csv"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    if "csv" in formats:
        path = os.path.join(out_dir, f"trade_area_competitors_{stamp}.csv")
        df.to_csv(path, index=False)
        files.append(path)

    if "xlsx" in formats:
        path = os.path.join(out_dir, f"trade_area_competitors_{stamp}.xlsx")
        df.to_excel(path, index=False)
        files.append(path)

    if "pdf" in formats:
        path = os.path.join(out_dir, f"trade_area_snapshot_{stamp}.pdf")
        _write_pdf_snapshot(df, no_website, by_category, spec, path)
        files.append(path)

    manifest = {
        "spec": spec,
        "competitor_count": len(df),
        "no_website_count": len(no_website),
        "no_website_pct": round(100 * len(no_website) / len(df), 1),
        "nearest_competitor_miles": float(df["distance_miles"].min()),
        "by_category": by_category.to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    files.append(manifest_path)

    return files


def _write_pdf_snapshot(df: pd.DataFrame, no_website: pd.DataFrame, by_category: pd.Series, spec: dict, path: str):
    """One-page trade-area snapshot: this is the document an SBA loan
    applicant or their lender actually reads, not the raw CSV."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = [
        Paragraph("Trade-Area Competitor Snapshot", styles["Title"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"Center: ({spec['center_lat']}, {spec['center_lon']})  |  "
                  f"Radius: {spec['radius_miles']} miles  |  Category: {spec['category']}",
                  styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>{len(df)}</b> competitors found in trade area", styles["Heading2"]),
        Paragraph(f"<b>{len(no_website)}</b> ({round(100*len(no_website)/len(df),1)}%) have no listed website",
                  styles["Normal"]),
        Paragraph(f"Nearest competitor: {df['distance_miles'].min():.2f} miles away", styles["Normal"]),
        Spacer(1, 0.3 * inch),
    ]

    rows = [["Category", "Count"]] + [[c, str(n)] for c, n in by_category.head(15).items()]
    t = Table(rows, colWidths=[3 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    doc.build(story)


def run_smoke_test(db_path: str):
    """Call this yourself with a real parquet path — not auto-run on import
    (same notebook-hijacking-sys.argv reason as service_lead_research_report.py)."""
    spec = {
        "center_lat": 39.9612,
        "center_lon": -82.9988,
        "radius_miles": 3,
        "category": "dental_clinic",
        "country": "US",
        "output_formats": ["csv", "pdf"],
    }
    preview = preview_trade_area(spec, db_path)
    print("Preview:", preview)
    if preview["competitor_count"] > 0:
        files = fulfil(spec, db_path, "./test_trade_area")
        print("Produced:", files)
    else:
        print("Zero competitors in this trade area for this spec — real finding, not an error.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] != "-f":
        run_smoke_test(sys.argv[1])
    else:
        print("No real file path given on the command line — skipping the smoke test.")
        print('In a notebook, call run_smoke_test("/path/to/business_database.parquet") yourself instead.')
