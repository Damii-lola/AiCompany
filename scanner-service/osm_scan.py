"""
Company Scrapper — OpenStreetMap Overpass discovery scanner
============================================================
A SECOND, independent free business-discovery source alongside the
Overture Maps scanner (scan_worker.py). Same TiDB shards, same
businesses table, same upsert semantics — this just walks a different
public dataset (OpenStreetMap) over the same US tile grid, tagging every
row it finds with source='osm' so it's never confused with Overture rows
(see database/schema.mysql.sql's businesses.source column).

WHY A SEPARATE SOURCE, NOT A REPLACEMENT: Overture and OSM are
independently-maintained datasets that miss different businesses — a
shop present in one is often absent from the other. Running both and
letting duplicates coexist (deduped later, at report-generation time,
when only ~50-300 candidates in one trade area need comparing — see
render-service's future enrichment pipeline) is far cheaper than trying
to cross-reference ~18.8M Overture rows against every OSM candidate
during bulk ingest, which would make ingest the slow part instead of a
one-time cost.

OVERPASS API: a free, keyless public API over OSM's database
(https://overpass-api.de/api/interpreter by default, overridable via
OVERPASS_URL for a mirror or a self-hosted instance). It has NO official
rate limit document but well-known fair-use norms: one query at a time
(never concurrent), a short pause between queries, and bounding boxes
sized to avoid the server's own timeout/memory limits — this script
follows all three (serial queries, SLEEP_BETWEEN_QUERIES pause, and the
same recursive quarter-and-retry pattern scan_worker.py's scan_region
uses for Overture, triggered here by an Overpass timeout/size error
instead of a DuckDB OOM).

WHAT'S QUERIED: every node and way tagged shop=*, amenity=*, office=*,
or craft=* that also has a name tag (filters out unnamed infrastructure
like benches or waste baskets, which carry amenity= tags too but aren't
businesses). `out center tags;` gives a usable lon/lat for ways (their
geometric center) without fetching full way geometry, which we don't
need.

TESTING NOTE: this sandbox's network can't reach overpass-api.de (same
allowlist restriction that blocked live TiDB/Render testing earlier in
this project) — this has been reviewed carefully against Overpass's
documented query syntax and response shape, and the record-building
logic was unit-tested against saved sample JSON (see
test_osm_scan_offline.py), but it has NOT been run against the live
Overpass API. Test with a small manual region via workflow_dispatch
before trusting the scheduled cron with it.

Required env vars: same DATABASE_URL_1 (.. _N) convention as
scan_worker.py. Tunable: OVERPASS_URL, OSM_TILE_DEG (smaller than
Overture's default since OSM data is far denser in cities and Overpass's
own server-side limits bite sooner), MAX_RUNTIME_SECONDS (GitHub
Actions' bounded-run pattern, same as scan_worker.py).
"""

import os
import sys
import time
import json
import requests

from scan_worker import (
    pg_connect, Lookups, pg_upsert_businesses, load_done_tiles, flush_done_tiles,
    discover_shards, assign_tiles_to_shards, US_REGIONS, _tiles_for_region,
)

sys.stdout.reconfigure(line_buffering=True)

OVERPASS_URL = os.environ.get("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
# Smaller than Overture's TILE_DEG=2 default — OSM's node density in a
# dense US metro routinely blows Overpass's default server-side memory/
# time limits at 2 degrees, where Overture's columnar parquet reads
# don't have the same problem. Still auto-quarters further on failure.
OSM_TILE_DEG = float(os.environ.get("OSM_TILE_DEG", "1"))
MIN_SPLIT_DEG = float(os.environ.get("OSM_MIN_SPLIT_DEG", "0.1"))
MAX_SPLIT_DEPTH = int(os.environ.get("OSM_MAX_SPLIT_DEPTH", "6"))
# Polite pause between successive Overpass requests — this is a shared
# public resource, not a paid API with a contracted rate limit, so this
# errs well on the slow side deliberately.
SLEEP_BETWEEN_QUERIES = float(os.environ.get("OSM_SLEEP_BETWEEN_QUERIES", "2.0"))
REQUEST_TIMEOUT = int(os.environ.get("OSM_REQUEST_TIMEOUT", "90"))
OSM_RELEASE = "osm"  # not a dated release like Overture's — OSM has no version concept here


def build_query(x0, y0, x1, y1):
    # Overpass bbox order is (south, west, north, east) = (y0, x0, y1, x1).
    bbox = f"{y0},{x0},{y1},{x1}"
    clauses = "\n  ".join(
        f'{kind}["{tag}"]["name"]({bbox});'
        for kind in ("node", "way")
        for tag in ("shop", "amenity", "office", "craft")
    )
    return f"""
[out:json][timeout:{REQUEST_TIMEOUT - 10}];
(
  {clauses}
);
out center tags;
""".strip()


def fetch_overpass(session, x0, y0, x1, y1, depth=0):
    """Queries one bbox, auto-quartering on a timeout/size error exactly
    like scan_worker.py's scan_region does for Overture's DuckDB OOM case
    — same reasoning, different failure mode (Overpass returns HTTP 504
    or a "runtime error" JSON body when a query is too big for its
    server-side limits, rather than raising a Python OOM exception).
    Returns a flat list of raw Overpass elements."""
    query = build_query(x0, y0, x1, y1)
    try:
        resp = session.post(OVERPASS_URL, data={"data": query}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 429:
            # Overpass's documented way of saying "slow down" — back off
            # hard and retry the SAME bbox once rather than splitting it
            # (splitting wouldn't help; the server is just busy).
            time.sleep(30)
            resp = session.post(OVERPASS_URL, data={"data": query}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "remark" in data and "error" in data.get("remark", "").lower():
            raise RuntimeError(f"Overpass runtime error: {data['remark']}")
        return data.get("elements", [])
    except Exception as e:
        width = x1 - x0
        if depth >= MAX_SPLIT_DEPTH or width <= MIN_SPLIT_DEG:
            print(f"    region ({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f}) still failing at the smallest "
                  f"split size ({width:.3f} deg, depth {depth}): {e}")
            raise
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        print(f"    region ({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f}) failed ({type(e).__name__}: {e}) "
              f"— splitting into 4 quadrants (depth {depth + 1}) and retrying each")
        elements = []
        for (sx0, sy0, sx1, sy1) in [(x0, y0, mx, my), (mx, y0, x1, my),
                                      (x0, my, mx, y1), (mx, my, x1, y1)]:
            time.sleep(SLEEP_BETWEEN_QUERIES)
            elements.extend(fetch_overpass(session, sx0, sy0, sx1, sy1, depth + 1))
        return elements


def element_to_record(el):
    """Maps one raw Overpass element to the same record dict shape
    pg_upsert_businesses expects from scan_worker.py's Overture path —
    see that function's norm()/rows.append() for the exact field list
    this must match."""
    tags = el.get("tags") or {}
    name = tags.get("name")
    if not name:
        return None

    if el["type"] == "node":
        lat, lon = el.get("lat"), el.get("lon")
    else:
        center = el.get("center") or {}
        lat, lon = center.get("lat"), center.get("lon")
    if lat is None or lon is None:
        return None

    category = tags.get("shop") or tags.get("amenity") or tags.get("office") or tags.get("craft")

    street_parts = [tags.get("addr:housenumber"), tags.get("addr:street")]
    address = " ".join(p for p in street_parts if p) or None

    social_parts = [tags.get(f"contact:{p}") for p in ("facebook", "instagram", "twitter")]
    social = ", ".join(p for p in social_parts if p) or None

    return {
        "id": f"osm_{el['type']}_{el['id']}",
        "name": name,
        "category": category,
        "category_extra": None,
        "address": address,
        "city": tags.get("addr:city"),
        "region": tags.get("addr:state") or tags.get("addr:province"),
        "country": "US",  # queried bbox is always within US_REGIONS
        "postcode": tags.get("addr:postcode"),
        "phone": tags.get("phone") or tags.get("contact:phone"),
        "website": tags.get("website") or tags.get("contact:website"),
        "email": tags.get("email") or tags.get("contact:email"),
        "social": social,
        "brand": tags.get("brand"),
        # OSM has no single universal "confidence" concept like Overture's
        # per-row ML confidence score — 0.7 is a fixed, deliberately
        # middling value (not invented data, just an honest "this came
        # from community-tagged OSM, not a confidence model") so it never
        # outranks or underranks Overture rows unfairly in anything that
        # sorts/filters on confidence.
        "confidence": 0.7,
        "operating_status": "open",  # the Overpass query already excludes disused:*-tagged elements
        "lon": lon,
        "lat": lat,
        "source": "osm",
    }


def run_shard_pass(shard_num, url, tiles, con_factory, session):
    pg = con_factory(url)
    lookups = Lookups(pg)
    done = load_done_tiles(pg, OSM_RELEASE)
    pending_tiles = []
    total_found = 0
    for i, (x0, y0, x1, y1) in enumerate(tiles, 1):
        tile_key = f"{OSM_RELEASE}:{x0}_{y0}"
        if tile_key in done:
            continue
        try:
            elements = fetch_overpass(session, x0, y0, x1, y1)
            records = [r for r in (element_to_record(el) for el in elements) if r]
            # Same NUL-byte protection scan_worker.py applies to Overture
            # free-text fields — OSM's community-entered tags are just as
            # capable of carrying a stray bad byte.
            records = [
                {k: (v.replace("\x00", "") if isinstance(v, str) else v) for k, v in r.items()}
                for r in records
            ]
            pg_upsert_businesses(pg, lookups, records)
            total_found += len(records)
            print(f"[shard {shard_num}] [{i}/{len(tiles)}] tile {tile_key}: {len(records)} OSM rows synced")
            pending_tiles.append((tile_key, OSM_RELEASE, len(records)))
            if len(pending_tiles) >= 20:
                flush_done_tiles(pg, pending_tiles)
        except Exception as e:
            print(f"[shard {shard_num}] tile {tile_key} failed even after splitting: {e} — will retry on a later run")
        time.sleep(SLEEP_BETWEEN_QUERIES)
    if pending_tiles:
        flush_done_tiles(pg, pending_tiles)
    pg.close()
    return total_found


def main():
    shards = discover_shards()
    shard_nums = ", ".join(str(n) for n, _ in shards)
    max_runtime = os.environ.get("MAX_RUNTIME_SECONDS")
    print(f"OSM Overpass discovery scanner starting. {len(shards)} shard database(s) configured "
          f"(shard numbers: {shard_nums})."
          + (f" Bounded run: exiting by ~{int(max_runtime)}s from now." if max_runtime else " Running indefinitely."))

    tiles = [t for (x0, y0, x1, y1) in US_REGIONS for t in _tiles_for_region(x0, y0, x1, y1, OSM_TILE_DEG)]
    # assign_tiles_to_shards returns a plain list of buckets BY POSITION,
    # not a dict keyed by shard number — zip with `shards` (same order
    # discover_shards sorted them in) to pair each bucket with its real
    # shard, exactly like scan_worker.py's run_one_pass does.
    buckets = assign_tiles_to_shards(tiles, len(shards))
    print(f"{len(tiles)} tiles total ({OSM_TILE_DEG}x{OSM_TILE_DEG} deg each, US + SBA territories only).")

    session = requests.Session()
    deadline = time.monotonic() + int(max_runtime) if max_runtime else None

    # Deliberately SERIAL across shards too (unlike scan_worker.py's
    # concurrent-per-shard threads) — Overpass is one shared public
    # server, and hitting it from several threads at once is exactly the
    # kind of load the fair-use norms this file's docstring describes are
    # meant to prevent, even though our own shard databases could easily
    # take concurrent writes.
    for (shard_num, url), shard_tiles in zip(shards, buckets):
        if deadline and time.monotonic() >= deadline:
            print("Runtime budget exhausted — stopping before the next shard.")
            break
        print(f"--- shard {shard_num}: {len(shard_tiles)} tiles ---")
        run_shard_pass(shard_num, url, shard_tiles, pg_connect, session)

    print("OSM Overpass pass complete (or runtime budget reached).")


if __name__ == "__main__":
    main()
