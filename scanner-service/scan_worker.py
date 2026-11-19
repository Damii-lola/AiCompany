"""
Company Scrapper — continuous scan worker (GitHub Actions / Render / any VM)
======================================================================
Replaces the old Kaggle-notebook scanner for this job specifically. This
was NEVER GPU work — DuckDB reading Overture Maps' S3 parquet is CPU/
memory/network bound, nothing here ever touched a GPU.

STORE: Prisma Postgres (direct psycopg2 connection — Prisma Postgres has
no REST API like Supabase did, so this talks straight Postgres). Schema
lives in database/schema.sql: a normalized schema (category/location/
brand/status stored as small integer foreign keys into lookup tables
instead of repeated text) built specifically to fit inside Prisma's free
tier at real scan volume, plus LZ4 TOAST compression on the remaining
free-text columns.

SCOPE: United States + the territories SBA actually lends in (Puerto
Rico, Guam, US Virgin Islands, American Samoa, Northern Mariana Islands).
Earlier this scanned the whole planet — ~88M+ rows, 40-50GB+ with
indexes, which doesn't fit ANY provider's free tier, Prisma's included.
SBA lending is a US program, so the other ~190 countries' data was never
going to be used by this product. Scoping to the US drops both the
compute (way fewer tiles to walk) and the storage (an order of magnitude
fewer rows) at the same time, which is the only lever that actually fixes
"too big" — compression and a smaller provider tier help on the margins,
but scope is what makes this fit at all.

Resumability lives entirely in Postgres (the scan_tiles table), so this
worker needs ZERO local disk state and can be killed/restarted at any
point with nothing lost — it just picks up whatever tiles aren't marked
done yet for the current release.

Operation-count discipline (Prisma's free tier metering, not just
storage, caps you — 200k ops/month): lookup tables (categories,
locations, brands, statuses) are loaded into memory ONCE per pass, not
queried per row or per tile. New lookup values seen in a batch are
inserted in one bulk statement per batch, not one per value. Tile-done
checks are loaded as a single set per pass instead of one query per tile.
businesses are upserted with execute_values in batches, not row by row.

Required env var: DATABASE_URL (a Prisma Postgres connection string —
either the direct db.prisma.io or pooled pooled.db.prisma.io string both
work for this worker's query pattern).
Tunable env vars (see .env.example): TILE_DEG, DUCKDB_THREADS,
DUCKDB_MEMORY_LIMIT, MIN_CONFIDENCE, SCAN_IDLE_SLEEP_SECONDS,
INGEST_BATCH_SIZE.
"""

import sys
import os
import json
import time
from datetime import datetime, timezone

import duckdb
import requests
import psycopg2
import psycopg2.extras

# Render/Actions' log stream isn't a real terminal, so Python block-buffers
# stdout by default instead of flushing per-line. Force line buffering so
# every print() below actually reaches the log in real time.
sys.stdout.reconfigure(line_buffering=True)

DATABASE_URL = os.environ["DATABASE_URL"]

# A fixed tile size can never be "safe" everywhere — a 2-degree box over
# open ocean and a 2-degree box over Manhattan are wildly different
# workloads. TILE_DEG only sets the STARTING grid; any individual region
# that still blows memory gets automatically quartered and retried (see
# scan_region below).
TILE_DEG = int(os.environ.get("TILE_DEG", "2"))
MIN_SPLIT_DEG = float(os.environ.get("MIN_SPLIT_DEG", "0.1"))  # stop subdividing below this size
MAX_SPLIT_DEPTH = int(os.environ.get("MAX_SPLIT_DEPTH", "8"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.0"))
THREADS = int(os.environ.get("DUCKDB_THREADS", "1"))
MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "300MB")
SCAN_IDLE_SLEEP_SECONDS = int(os.environ.get("SCAN_IDLE_SLEEP_SECONDS", str(6 * 3600)))
INGEST_BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "500"))
FALLBACK_RELEASE = "2026-10-21.0"  # only used if the live STAC lookup fails

# United States bounding box (CONUS + Alaska + Hawaii + Puerto Rico + US
# Virgin Islands) plus small explicit boxes for the three Pacific
# territories that fall way outside that box. country='US' is still
# applied in SQL on every box, so e.g. the slivers of Canada/Mexico/
# Bahamas/Marshall Islands/Palau that fall inside these boxes get
# filtered out there, not by the geography alone.
US_REGIONS = [
    (-180.0, 14.0, -64.0, 72.0),    # CONUS, Alaska, Hawaii, PR, USVI
    (144.5, 13.0, 145.2, 15.3),     # Guam
    (144.8, 14.0, 146.2, 20.6),     # Northern Mariana Islands
    (-170.9, -14.6, -169.3, -14.0),  # American Samoa
]


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def pg_connect():
    con = psycopg2.connect(DATABASE_URL)
    con.autocommit = True
    return con


class Lookups:
    """categories/locations/brands/statuses, loaded once per pass and kept
    in memory for the whole run — this is what keeps per-tile Postgres
    traffic to a handful of statements instead of a query per row."""

    def __init__(self, pg):
        self.pg = pg
        self.categories = self._load("categories", "name")
        self.locations = self._load_locations()
        self.brands = self._load("brands", "name")
        self.statuses = self._load("statuses", "name")

    def _load(self, table, col):
        with self.pg.cursor() as cur:
            cur.execute(f"SELECT {col}, id FROM {table}")
            return dict(cur.fetchall())

    def _load_locations(self):
        with self.pg.cursor() as cur:
            cur.execute("SELECT city, region, country, id FROM locations")
            return {(c, r, k): i for (c, r, k, i) in cur.fetchall()}

    def _ensure_simple(self, table, col, cache, values):
        """Bulk-inserts any values not already cached, in ONE statement,
        then merges the new ids back into the cache. values may contain
        None/''/duplicates — all filtered before hitting Postgres."""
        # isinstance check, not just truthiness: a stray float (NaN or
        # otherwise) landing in this set alongside real strings is exactly
        # what makes sorted({...}) raise "'<' not supported between
        # instances of 'float' and 'str'" — defends this even if some other
        # caller ever skips norm().
        new_values = sorted({v for v in values if isinstance(v, str) and v and v not in cache})
        if not new_values:
            return
        with self.pg.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {table} ({col}) VALUES %s ON CONFLICT ({col}) DO NOTHING",
                [(v,) for v in new_values],
            )
            cur.execute(f"SELECT {col}, id FROM {table} WHERE {col} = ANY(%s)", (new_values,))
            cache.update(cur.fetchall())

    def ensure_categories(self, values):
        self._ensure_simple("categories", "name", self.categories, values)

    def ensure_brands(self, values):
        self._ensure_simple("brands", "name", self.brands, values)

    def ensure_statuses(self, values):
        self._ensure_simple("statuses", "name", self.statuses, values)

    def ensure_locations(self, triples):
        """triples: iterable of (city, region, country) with '' for missing
        city/region (never None — the unique constraint treats every null
        as distinct, so '' is the real "no value" here, same convention as
        the locations table itself)."""
        new_triples = sorted({
            t for t in triples
            if all(isinstance(x, str) for x in t) and t[2] and t not in self.locations
        })
        if not new_triples:
            return
        with self.pg.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO locations (city, region, country) VALUES %s "
                "ON CONFLICT (city, region, country) DO NOTHING",
                new_triples,
            )
            # Row-comparison against an array of tuples isn't something
            # psycopg2/Postgres do directly ("= ANY(%s)" fails with a
            # DatatypeMismatch for composite values) — zip the three
            # columns back into rows with unnest() and join on that instead.
            cities = [t[0] for t in new_triples]
            regions = [t[1] for t in new_triples]
            countries = [t[2] for t in new_triples]
            cur.execute(
                """
                SELECT l.city, l.region, l.country, l.id
                FROM locations l
                JOIN (
                    SELECT unnest(%s::text[]) AS city, unnest(%s::text[]) AS region, unnest(%s::text[]) AS country
                ) t ON l.city = t.city AND l.region = t.region AND l.country = t.country
                """,
                (cities, regions, countries),
            )
            for (c, r, k, i) in cur.fetchall():
                self.locations[(c, r, k)] = i


def pg_upsert_businesses(pg, lookups, records):
    """Resolves category/location/brand/status text to integer ids using
    the in-memory lookup cache (bulk-inserting anything new first), then
    upserts the real business rows in batches with execute_values."""
    if not records:
        return

    def norm(v):
        # Deliberately stricter than "v if v else ''": a bare float NaN is
        # truthy in Python (bool(float('nan')) is True) and would otherwise
        # sail through here as a real lookup value, land in the same set as
        # real category/status strings, and blow up sorted({...}) with
        # "'<' not supported between instances of 'float' and 'str'" the
        # moment a float and a string land in the same set together.
        return v if isinstance(v, str) and v else ""

    lookups.ensure_categories(norm(r.get("category")) for r in records)
    lookups.ensure_categories(norm(r.get("category_extra")) for r in records)
    lookups.ensure_brands(norm(r.get("brand")) for r in records)
    lookups.ensure_statuses(norm(r.get("operating_status")) for r in records)
    lookups.ensure_locations(
        (norm(r.get("city")), norm(r.get("region")), norm(r.get("country")) or "US")
        for r in records
    )

    now = _now_iso()
    rows = []
    for r in records:
        country = norm(r.get("country")) or "US"
        loc_key = (norm(r.get("city")), norm(r.get("region")), country)
        rows.append((
            r["id"], r.get("name"),
            lookups.categories.get(norm(r.get("category"))),
            lookups.categories.get(norm(r.get("category_extra"))),
            r.get("address"),
            lookups.locations.get(loc_key),
            r.get("postcode"), r.get("phone"), r.get("website"), r.get("email"), r.get("social"),
            lookups.brands.get(norm(r.get("brand"))),
            r.get("confidence"),
            lookups.statuses.get(norm(r.get("operating_status"))),
            r.get("lon"), r.get("lat"),
            now,
        ))

    with pg.cursor() as cur:
        for i in range(0, len(rows), INGEST_BATCH_SIZE):
            batch = rows[i:i + INGEST_BATCH_SIZE]
            psycopg2.extras.execute_values(cur, """
                INSERT INTO businesses (
                    id, name, category_id, category_extra_id, address, location_id,
                    postcode, phone, website, email, social, brand_id, confidence,
                    status_id, lon, lat, last_seen_at
                ) VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    name = excluded.name, category_id = excluded.category_id,
                    category_extra_id = excluded.category_extra_id, address = excluded.address,
                    location_id = excluded.location_id, postcode = excluded.postcode,
                    phone = excluded.phone, website = excluded.website, email = excluded.email,
                    social = excluded.social, brand_id = excluded.brand_id,
                    confidence = excluded.confidence, status_id = excluded.status_id,
                    lon = excluded.lon, lat = excluded.lat, last_seen_at = excluded.last_seen_at
            """, batch)


def load_done_tiles(pg, release):
    """One query for the WHOLE pass instead of one per tile — this is the
    change that keeps tile-done checks off Prisma's metered-operations
    budget almost entirely."""
    with pg.cursor() as cur:
        cur.execute("SELECT tile_key FROM scan_tiles WHERE release = %s", (release,))
        return {row[0] for row in cur.fetchall()}


def flush_done_tiles(pg, pending):
    """pending: list of (tile_key, release, rows_found) tuples, batched and
    flushed together rather than one insert per tile."""
    if not pending:
        return
    with pg.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO scan_tiles (tile_key, release, rows_found, scanned_at) VALUES %s
            ON CONFLICT (tile_key) DO UPDATE SET
                rows_found = excluded.rows_found, scanned_at = excluded.scanned_at
        """, [(k, r, n, _now_iso()) for (k, r, n) in pending])
    pending.clear()


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
    slightly release to release."""
    for expr, needed_col in candidates:
        if needed_col in cols:
            return expr
    return "NULL"


def scan_region(con, pg, lookups, select, cols, x0, y0, x1, y1, depth=0):
    """Queries one lon/lat bounding box, restricted to country='US' so we
    never pull (and never store) non-US places that happen to fall inside
    one of the US_REGIONS boxes. If it fails (almost always an OOM on a
    dense region) and the box is still bigger than MIN_SPLIT_DEG and we
    haven't hit MAX_SPLIT_DEPTH, it quarters the box and recurses on each
    quadrant. Returns the real row count found (0 is a legitimate answer
    for an empty region — never used to mean 'failed')."""
    where = (
        f"bbox.xmin >= {x0} AND bbox.xmin < {x1} AND bbox.ymin >= {y0} AND bbox.ymin < {y1} "
        f"AND country = 'US'"
    )
    if "confidence" in cols and MIN_CONFIDENCE > 0:
        where += f" AND confidence >= {MIN_CONFIDENCE}"
    try:
        df = con.execute(f"SELECT {select} FROM places WHERE {where}").fetchdf()
        n = len(df)
        if n > 0:
            # json round-trip, not df.where(df.notnull(), None).to_dict():
            # pandas frequently reverts an assigned None right back to NaN
            # on a float64 column (a well-known pandas quirk), so that
            # version was silently leaking real float('nan') values into
            # category/status/confidence fields — which then crashed
            # sorted({...}) in Lookups the moment a NaN and a real string
            # landed in the same set. JSON serialization has no such
            # quirk: NaN always becomes JSON null, which json.loads() always
            # turns back into a real Python None, for every dtype.
            records = json.loads(df.to_json(orient="records"))
            pg_upsert_businesses(pg, lookups, records)
        return n
    except Exception as e:
        width = x1 - x0
        if depth >= MAX_SPLIT_DEPTH or width <= MIN_SPLIT_DEG:
            print(f"    region ({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f}) still failing at the smallest "
                  f"split size ({width:.3f} deg, depth {depth}): {e}")
            raise
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        print(f"    region ({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f}) failed ({type(e).__name__}: {e}) "
              f"— splitting into 4 quadrants (depth {depth + 1}) and retrying each")
        total = 0
        for (sx0, sy0, sx1, sy1) in [(x0, y0, mx, my), (mx, y0, x1, my),
                                      (x0, my, mx, y1), (mx, my, x1, y1)]:
            total += scan_region(con, pg, lookups, select, cols, sx0, sy0, sx1, sy1, depth + 1)
        return total


def _tiles_for_region(x0, y0, x1, y1, step):
    x = x0
    while x < x1:
        y = y0
        while y < y1:
            yield (x, y, min(x + step, x1), min(y + step, y1))
            y += step
        x += step


def run_one_pass(release, pg, deadline=None):
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

    tiles = [t for (x0, y0, x1, y1) in US_REGIONS for t in _tiles_for_region(x0, y0, x1, y1, TILE_DEG)]
    print(f"Pass against release {release}: {len(tiles)} tiles total ({TILE_DEG}x{TILE_DEG} deg each, "
          f"US + SBA territories only).")

    lookups = Lookups(pg)
    done_tiles = load_done_tiles(pg, release)  # ONE query for the whole pass
    pending_done = []
    did_any_work = False

    for i, (x0, y0, x1, y1) in enumerate(tiles, 1):
        if deadline is not None and time.monotonic() >= deadline:
            print(f"  runtime cap reached mid-pass at tile {i}/{len(tiles)} — stopping here, "
                  f"cleanly, nothing lost (unprocessed tiles just aren't marked done yet).")
            flush_done_tiles(pg, pending_done)
            break

        tile_key = f"{release}:{x0}_{y0}"
        if tile_key in done_tiles:
            continue
        did_any_work = True

        n = None
        for attempt in range(3):
            try:
                n = scan_region(con, pg, lookups, select, cols, x0, y0, x1, y1)
                break
            except Exception as e:
                print(f"  tile {tile_key} attempt {attempt + 1}/3 failed even after splitting: {e}")
                time.sleep(5 * (attempt + 1))
        if n is None:
            print(f"  SKIPPED {tile_key} after 3 failed attempts — "
                  f"not marked done, will retry on a later loop iteration.")
            continue

        pending_done.append((tile_key, release, n))
        if len(pending_done) >= 20:
            flush_done_tiles(pg, pending_done)
        print(f"[{i}/{len(tiles)}] tile {tile_key}: {n:,} US rows synced to Postgres")

    flush_done_tiles(pg, pending_done)
    con.close()
    return did_any_work


def main():
    # MAX_RUNTIME_SECONDS is for a bounded host that re-triggers this
    # process on its own schedule instead of letting it run forever — a
    # GitHub Actions job (6-hour hard cap per job) being the real case this
    # was added for. Unset (the default) means run forever (Render, a VM).
    max_runtime = os.environ.get("MAX_RUNTIME_SECONDS")
    deadline = time.monotonic() + int(max_runtime) if max_runtime else None

    print("Company Scrapper scan worker starting — US + SBA territories only, "
          "no GPU required, no local disk state kept."
          + (f" Bounded run: exiting by ~{int(max_runtime)}s from now." if deadline else " Running indefinitely."))

    pg = pg_connect()
    while True:
        if deadline and time.monotonic() >= deadline:
            print("MAX_RUNTIME_SECONDS reached — exiting cleanly so the scheduler can re-trigger. "
                  "Nothing lost: progress lives in Postgres's scan_tiles table, not in this process.")
            pg.close()
            return

        release = latest_release()
        try:
            did_work = run_one_pass(release, pg, deadline)
        except Exception as e:
            print(f"Pass against {release} failed with an unhandled error: {e}. Retrying in 5 minutes.")
            if pg.closed:
                pg = pg_connect()
            _sleep_respecting_deadline(300, deadline)
            continue

        if not did_work:
            if deadline:
                print(f"Full pass against release {release} already complete — "
                      f"nothing left to do this run, exiting cleanly.")
                pg.close()
                return
            print(f"Full pass against release {release} already complete. "
                  f"Sleeping {SCAN_IDLE_SLEEP_SECONDS}s, then checking for a newer release...")
            time.sleep(SCAN_IDLE_SLEEP_SECONDS)
        # loop again regardless — either there's leftover work (retried
        # tiles from a prior failure) or we just finished sleeping and
        # latest_release() may now return something new


def _sleep_respecting_deadline(seconds, deadline):
    if deadline is None:
        time.sleep(seconds)
        return
    remaining = deadline - time.monotonic()
    time.sleep(max(0, min(seconds, remaining)))


if __name__ == "__main__":
    main()
