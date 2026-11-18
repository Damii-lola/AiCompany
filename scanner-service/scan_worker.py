"""
Company Scrapper — continuous scan worker (Render Background Worker)
======================================================================
Replaces the old Kaggle-notebook scanner for this job specifically. This
was NEVER GPU work — DuckDB reading Overture Maps' S3 parquet is CPU/
memory/network bound, nothing here ever touched a GPU. Kaggle was only
ever used for (a) free compute and (b) Kaggle Datasets doubling as
resumable checkpoint storage. Neither reason holds anymore: Supabase is
now the live store, so resumability lives there too (the scan_tiles
table from supabase/schema.sql), which means this worker needs ZERO
local disk state and can be killed/restarted by Render at any point with
nothing lost — it just picks up whatever tiles aren't marked done yet.

Kaggle still matters for exactly the things that genuinely need a GPU:
training the template model (train_the_ai.py) and deep on-demand natural-
language research sessions with the real trained model
(company_scrapper_ai.py). This worker does neither — it's pure
deterministic data retrieval, same reasoning as why Company Scrapper's
Render API never needed an LLM either.

Loop, forever:
  1. Find the current Overture release.
  2. Walk the world in TILE_DEG x TILE_DEG tiles. For each tile not yet
     marked done (in Supabase's scan_tiles, keyed by release+tile so a
     newer release's tiles are automatically "not done" with no reset
     step ever needed): query it straight from S3 via DuckDB, upsert the
     real rows into Supabase's businesses table, mark the tile done.
  3. Once every tile for the current release is done, that's a full pass:
     sleep SCAN_IDLE_SLEEP_SECONDS, then loop back to step 1 — which will
     naturally pick up a newer release if Overture has published one
     (they publish roughly monthly; this paces itself to that, it never
     busy-loops hammering S3 for nothing).

Required env vars: SUPABASE_URL, SUPABASE_SERVICE_KEY.
Tunable env vars (see .env.example): TILE_DEG, DUCKDB_THREADS,
DUCKDB_MEMORY_LIMIT, MIN_CONFIDENCE, SCAN_IDLE_SLEEP_SECONDS,
INGEST_BATCH_SIZE.
"""

import os
import json
import time
from datetime import datetime, timezone

import duckdb
import requests

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

TILE_DEG = int(os.environ.get("TILE_DEG", "10"))  # smaller than the old Kaggle version's 15 —
                                                     # keeps each tile's memory footprint modest
                                                     # for a modest Render worker plan's RAM
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.0"))
THREADS = int(os.environ.get("DUCKDB_THREADS", "2"))
MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "1536MB")
SCAN_IDLE_SLEEP_SECONDS = int(os.environ.get("SCAN_IDLE_SLEEP_SECONDS", str(6 * 3600)))
INGEST_BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "500"))
FALLBACK_RELEASE = "2026-10-21.0"  # only used if the live STAC lookup fails

_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def supabase_upsert_businesses(records):
    """Real upsert via PostgREST's merge-duplicates conflict resolution —
    same semantics as the Node service's on_conflict('id') upsert, just
    called directly since this worker already holds the Supabase key
    (it's Render-side infra, not Kaggle, so that's consistent with the
    'only Render holds Supabase credentials' rule, not a violation of it)."""
    if not records:
        return
    headers = dict(_HEADERS)
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    for i in range(0, len(records), INGEST_BATCH_SIZE):
        batch = records[i:i + INGEST_BATCH_SIZE]
        for b in batch:
            b["last_seen_at"] = _now_iso()
        r = requests.post(f"{SUPABASE_URL}/rest/v1/businesses", headers=headers, json=batch, timeout=60)
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(f"Supabase upsert failed (HTTP {r.status_code}): {r.text[:400]}")


def supabase_tile_done(tile_key):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/scan_tiles", headers=_HEADERS,
        params={"tile_key": f"eq.{tile_key}", "select": "tile_key"}, timeout=15,
    )
    r.raise_for_status()
    return len(r.json()) > 0


def supabase_mark_tile_done(tile_key, release, n):
    headers = dict(_HEADERS)
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    row = {"tile_key": tile_key, "release": release, "rows_found": n, "scanned_at": _now_iso()}
    r = requests.post(f"{SUPABASE_URL}/rest/v1/scan_tiles", headers=headers, json=[row], timeout=15)
    if r.status_code not in (200, 201, 204):
        raise RuntimeError(f"Failed to mark tile done (HTTP {r.status_code}): {r.text[:300]}")


def latest_release():
    try:
        r = requests.get("https://stac.overturemaps.org/catalog.json", timeout=30).json()
        return r["latest"]
    except Exception as e:
        print(f"STAC release lookup failed ({e}) — using fallback {FALLBACK_RELEASE}.")
        return FALLBACK_RELEASE


def _pick(cols, *candidates):
    """Picks the first column expression whose source column actually
    exists in this release's schema — Overture's exact field names drift
    slightly release to release, and this must never silently emit NULL
    for a column that's really there under a different name without at
    least trying the known alternates first."""
    for expr, needed_col in candidates:
        if needed_col in cols:
            return expr
    return "NULL"


def run_one_pass(release):
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    con.execute(f"SET threads={THREADS}; SET memory_limit='{MEMORY_LIMIT}';")
    src = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
    con.execute(f"CREATE VIEW places AS SELECT * FROM read_parquet('{src}', hive_partitioning=1);")
    cols = {r[0] for r in con.execute("DESCRIBE places").fetchall()}

    select = f"""
      id,
      {_pick(cols, ("names.primary", "names"))} AS name,
      {_pick(cols, ("categories.primary", "categories"))} AS category,
      {_pick(cols, ("categories.alternate", "categories"))} AS category_extra,
      {_pick(cols, ("addresses[1].freeform", "addresses"))} AS address,
      {_pick(cols, ("addresses[1].locality", "addresses"))} AS city,
      {_pick(cols, ("addresses[1].region", "addresses"))} AS region,
      {_pick(cols, ("addresses[1].postcode", "addresses"))} AS postcode,
      {_pick(cols, ("addresses[1].country", "addresses"))} AS country,
      {_pick(cols, ("phones[1]", "phones"))} AS phone,
      {_pick(cols, ("websites[1]", "websites"))} AS website,
      {_pick(cols, ("emails[1]", "emails"))} AS email,
      {_pick(cols, ("socials[1]", "socials"))} AS social,
      {_pick(cols, ("brand.names.primary", "brand"))} AS brand,
      {_pick(cols, ("confidence", "confidence"))} AS confidence,
      {_pick(cols, ("operating_status", "operating_status"))} AS operating_status,
      bbox.xmin AS lon,
      bbox.ymin AS lat
    """

    tiles = [(x, y) for x in range(-180, 180, TILE_DEG) for y in range(-90, 90, TILE_DEG)]
    print(f"Pass against release {release}: {len(tiles)} tiles total ({TILE_DEG}x{TILE_DEG} deg each).")
    did_any_work = False

    for i, (x, y) in enumerate(tiles, 1):
        tile_key = f"{release}:{x}_{y}"
        if supabase_tile_done(tile_key):
            continue
        did_any_work = True

        where = (f"bbox.xmin >= {x} AND bbox.xmin < {x + TILE_DEG} "
                 f"AND bbox.ymin >= {y} AND bbox.ymin < {y + TILE_DEG}")
        if "confidence" in cols and MIN_CONFIDENCE > 0:
            where += f" AND confidence >= {MIN_CONFIDENCE}"

        df = None
        for attempt in range(4):
            try:
                df = con.execute(f"SELECT {select} FROM places WHERE {where}").fetchdf()
                break
            except Exception as e:
                print(f"  tile {tile_key} attempt {attempt + 1}/4 failed: {e}")
                time.sleep(5 * (attempt + 1))
        if df is None:
            print(f"  SKIPPED {tile_key} after 4 failed attempts — "
                  f"not marked done, will retry on a later loop iteration.")
            continue

        n = len(df)
        if n > 0:
            records = json.loads(df.to_json(orient="records"))  # NaN -> real null, safely
            supabase_upsert_businesses(records)
        supabase_mark_tile_done(tile_key, release, n)
        print(f"[{i}/{len(tiles)}] tile {tile_key}: {n:,} rows synced to Supabase")

    con.close()
    return did_any_work


def main():
    print("Company Scrapper scan worker starting — Render background worker, "
          "no GPU required, no local disk state kept.")
    while True:
        release = latest_release()
        try:
            did_work = run_one_pass(release)
        except Exception as e:
            # Never let one bad pass kill the worker forever — Render will
            # restart on a hard crash anyway, but a transient S3/network
            # blip shouldn't require manual intervention.
            print(f"Pass against {release} failed with an unhandled error: {e}. Retrying in 5 minutes.")
            time.sleep(300)
            continue

        if not did_work:
            print(f"Full pass against release {release} already complete. "
                  f"Sleeping {SCAN_IDLE_SLEEP_SECONDS}s, then checking for a newer release...")
            time.sleep(SCAN_IDLE_SLEEP_SECONDS)
        # loop again regardless — either there's leftover work (retried
        # tiles from a prior failure) or we just finished sleeping and
        # latest_release() may now return something new


if __name__ == "__main__":
    main()
