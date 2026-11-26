"""
Company Scrapper — continuous scan worker (GitHub Actions / Render / any VM)
======================================================================
Replaces the old Kaggle-notebook scanner for this job specifically. This
was NEVER GPU work — DuckDB reading Overture Maps' S3 parquet is CPU/
memory/network bound, nothing here ever touched a GPU.

STORE: TiDB Cloud Starter (direct PyMySQL connection — TiDB speaks the
MySQL wire protocol, not Postgres). Moved off Prisma Postgres entirely:
Prisma's 2GB/workspace free-tier cap was already being exceeded by the
live data, and scaling it further meant more separate email sign-ups,
which ran out. TiDB Cloud Starter gives up to 5 free instances PER
ACCOUNT (no card needed), 5GiB storage + 50M request units/month each —
25GB total under one login, no extra emails needed. Schema lives in
database/schema.mysql.sql: same normalized design as before (category/
location/brand/status stored as small integer foreign keys into lookup
tables instead of repeated text), rewritten to MySQL/TiDB dialect — see
that file's header for exactly what changed and why (no stored
functions/procedures or spatial types on TiDB, so the old
businesses_within_radius/businesses_trade_area_summary Postgres functions
don't exist here anymore; render-service/lib/trade-area.js does that math
in application code instead).

SHARDING: a single free TiDB Cloud Starter instance caps out at 5GiB
storage and 50M request units/month, which the US+territories backfill
can still exceed before finishing one pass depending on how many
instances are configured. This worker splits the work across as many free
TiDB instances as you give it — set DATABASE_URL_1, DATABASE_URL_2,
DATABASE_URL_3, ... (any number of them, up to the 5 a single TiDB Cloud
account gives you for free) as secrets and this worker:
  1. splits the FULL tile list evenly across however many shards you
     configured (round-robin by tile index, not a contiguous block —
     see assign_tiles_to_shards below for why),
  2. scans every shard's share CONCURRENTLY, one worker thread per shard,
     each with its own DuckDB connection and its own Postgres connection
     — all N databases are being written to at the same time, not one
     after another.
Each shard is a fully normal, independent copy of database/schema.mysql.sql —
apply the schema to EVERY shard database before adding its
DATABASE_URL_N secret. Lookup-table ids (category/location/brand/status)
are NOT consistent across shards, but that's fine: every join always
happens within one shard's own connection, never across shards.

IMPORTANT: the shard COUNT has to stay fixed for an entire backfill pass.
Tile-to-shard assignment is computed fresh from whatever DATABASE_URL_N
secrets exist right now, so adding or removing a shard mid-pass reshuffles
which shard owns which tile and would re-scan data into the wrong place.
Decide how many shards you want, set all of them before you start (or
before the next scheduled run picks up), and leave the count alone until
a full pass finishes. This is a one-time backfill job anyway (see below) —
once it's done, a shard count never needs to change again for it.
A bare DATABASE_URL (no numeric suffix) still works too, as a single
implicit shard — nothing above requires multiple databases.

SCOPE: United States + the territories SBA actually lends in (Puerto
Rico, Guam, US Virgin Islands, American Samoa, Northern Mariana Islands).
Earlier this scanned the whole planet — ~88M+ rows, 40-50GB+ with
indexes, which doesn't fit ANY provider's free tier, Prisma's included.
SBA lending is a US program, so the other ~190 countries' data was never
going to be used by this product. Scoping to the US drops both the
compute (way fewer tiles to walk) and the storage (an order of magnitude
fewer rows) at the same time, which is the only lever that actually fixes
"too big" — sharding and scope both help, but scope is what makes any of
this possible at all.

Resumability lives entirely in Postgres (each shard's own scan_tiles
table), so this worker needs ZERO local disk state and can be killed/
restarted at any point with nothing lost — it just picks up whatever
tiles aren't marked done yet, per shard, for the current release. This
scanner's job is a one-time backfill ("find it and add it once") — a
separate, future, targeted "updater" script is the planned place for
checking whether a specific already-known business has since changed
(ownership transfer, closed down, moved); this worker intentionally never
overwrites a business it's already seen (see pg_upsert_businesses below).

Operation-count discipline (Prisma's free tier metering, not just
storage, caps you — 200k ops/month PER shard): lookup tables (categories,
locations, brands, statuses) are loaded into memory ONCE per shard per
pass, not queried per row or per tile. New lookup values seen in a batch
are inserted in one bulk statement per batch, not one per value. Tile-done
checks are loaded as a single set per shard per pass instead of one query
per tile. businesses are upserted with execute_values in batches, not row
by row.

Required env vars: DATABASE_URL_1 (and DATABASE_URL_2, DATABASE_URL_3,
... for more shards), or a bare DATABASE_URL for a single-database setup.
Tunable env vars (see .env.example): TILE_DEG, DUCKDB_THREADS,
DUCKDB_MEMORY_LIMIT, MIN_CONFIDENCE, SCAN_IDLE_SLEEP_SECONDS,
INGEST_BATCH_SIZE. Note DUCKDB_MEMORY_LIMIT applies PER SHARD THREAD and
all shards run at once, so total memory use is roughly
(number of shards) x DUCKDB_MEMORY_LIMIT — lower it if you configure a lot
of shards on a memory-capped runner.
"""

import sys
import os
import re
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from urllib.parse import urlparse, unquote, parse_qs

import duckdb
import requests
import pymysql
import pymysql.cursors

# Render/Actions' log stream isn't a real terminal, so Python block-buffers
# stdout by default instead of flushing per-line. Force line buffering so
# every print() below actually reaches the log in real time — including
# from the shard worker threads below, since prints from multiple threads
# interleave but each line is still written atomically.
sys.stdout.reconfigure(line_buffering=True)

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
    # Despite the name (kept for a minimal diff against every call site),
    # this now returns a real Python datetime, not an ISO string. A
    # Postgres TIMESTAMPTZ column happily parsed the old ISO-with-"+00:00"
    # string, but MySQL/TiDB's TIMESTAMP columns expect 'YYYY-MM-DD
    # HH:MM:SS' — PyMySQL serializes a real datetime object to that format
    # correctly regardless of tzinfo (it just prints the already-UTC field
    # values), so passing the object itself sidesteps the format mismatch
    # entirely rather than hand-formatting a string.
    return datetime.now(timezone.utc)


def discover_shards():
    """Finds every DATABASE_URL_<N> secret that's set (DATABASE_URL_1,
    DATABASE_URL_2, ...) and returns them as a sorted list of
    (shard_num, url) tuples. Falls back to a bare DATABASE_URL (no suffix)
    as a single implicit shard 1 if no numbered secrets are set at all, so
    a single-database setup works with zero config changes."""
    shards = []
    pattern = re.compile(r"^DATABASE_URL_(\d+)$")
    for key, value in os.environ.items():
        m = pattern.match(key)
        if m and value:
            shards.append((int(m.group(1)), value))
    shards.sort(key=lambda t: t[0])
    if not shards:
        bare = os.environ.get("DATABASE_URL")
        if not bare:
            raise RuntimeError(
                "No database configured. Set DATABASE_URL_1 (and DATABASE_URL_2, "
                "DATABASE_URL_3, ... for more shards), or a bare DATABASE_URL for a "
                "single-database setup."
            )
        shards = [(1, bare)]
    return shards


def _parse_connection_url(url):
    """TiDB Cloud connection strings look like mysql://user:pass@host:4000/db
    (optionally with a trailing ?ssl=false for a local, SSL-less MySQL/
    MariaDB test instance — see database/schema.mysql.sql's header). PyMySQL
    takes host/port/user/password/database as separate kwargs rather than a
    single URL, so this parses it apart the same way
    render-service/lib/db.js's parseConnectionUrl does on the Node side."""
    u = urlparse(url)
    qs = parse_qs(u.query)
    wants_ssl = (qs.get("ssl", ["true"])[0]) != "false"
    return {
        "host": u.hostname,
        "port": u.port or 4000,  # TiDB Cloud's default port
        "user": unquote(u.username or ""),
        "password": unquote(u.password or ""),
        "database": u.path.lstrip("/"),
        # TiDB Cloud's gateway uses a publicly-trusted cert, so an empty
        # dict (PyMySQL's "use TLS, verify against the system CA bundle")
        # is the right default — no custom CA bundle needed, unlike the old
        # Prisma Postgres setup which needed rejectUnauthorized:false for
        # its internal routing cert.
        "ssl": {} if wants_ssl else None,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.Cursor,
    }


# NOTE: every `pg`-prefixed name below (pg_connect, pg_upsert_businesses,
# the `pg` variable) is a holdover name from the Postgres/Prisma era — left
# as-is to keep this diff focused on the actual driver/SQL dialect change
# rather than a cosmetic rename. All of them are real TiDB/PyMySQL
# connections now, not Postgres.
def pg_connect(url):
    con = pymysql.connect(**_parse_connection_url(url))
    con.autocommit(True)
    return con


class Lookups:
    """categories/locations/brands/statuses, loaded once per shard per
    pass and kept in memory for the whole run — this is what keeps
    per-tile Postgres traffic to a handful of statements instead of a
    query per row. One Lookups instance per shard connection — the ids it
    caches are only meaningful against that one shard's own tables."""

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
        None/''/duplicates — all filtered before hitting the database."""
        # isinstance check, not just truthiness: a stray float (NaN or
        # otherwise) landing in this set alongside real strings is exactly
        # what makes sorted({...}) raise "'<' not supported between
        # instances of 'float' and 'str'" — defends this even if some other
        # caller ever skips norm().
        new_values = sorted({v for v in values if isinstance(v, str) and v and v not in cache})
        if not new_values:
            return
        with self.pg.cursor() as cur:
            # INSERT IGNORE skips rows that would violate the unique(col)
            # constraint instead of erroring — the MySQL/TiDB equivalent of
            # Postgres's ON CONFLICT DO NOTHING. executemany with a single
            # VALUES (%s) tuple is PyMySQL's bulk-insert form (it rewrites
            # this into one real multi-row INSERT, same as
            # psycopg2.extras.execute_values did).
            cur.executemany(f"INSERT IGNORE INTO {table} ({col}) VALUES (%s)", [(v,) for v in new_values])
            cur.execute(f"SELECT {col}, id FROM {table} WHERE {col} IN %s", (tuple(new_values),))
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
            cur.executemany(
                "INSERT IGNORE INTO locations (city, region, country) VALUES (%s, %s, %s)",
                new_triples,
            )
            # MySQL/TiDB has no clean row-value IN-list (the unnest()-joined
            # exact-triple match Postgres used) — fetch every row matching
            # ANY of the three columns independently (a superset; locations
            # is a small table, so over-fetching here is cheap) and filter
            # down to exact triple matches in Python instead of in SQL.
            cities = tuple({t[0] for t in new_triples})
            regions = tuple({t[1] for t in new_triples})
            countries = tuple({t[2] for t in new_triples})
            cur.execute(
                "SELECT city, region, country, id FROM locations "
                "WHERE city IN %s AND region IN %s AND country IN %s",
                (cities, regions, countries),
            )
            wanted = set(new_triples)
            for (c, r, k, i) in cur.fetchall():
                if (c, r, k) in wanted:
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
            r.get("source") or "overture",
            r.get("basic_category"), r.get("category_hierarchy"),
            now,
        ))

    with pg.cursor() as cur:
        for i in range(0, len(rows), INGEST_BATCH_SIZE):
            batch = rows[i:i + INGEST_BATCH_SIZE]
            # ON DUPLICATE KEY UPDATE only bumps last_seen_at, nothing else
            # — a business already in the table is left alone (its name/
            # category/address/etc. never get silently overwritten by a
            # later re-scan). That's by design: this scanner's job is "find
            # it and add it once," not "keep re-verifying it." Real-world
            # changes to an existing business (ownership transfer, closed
            # down, moved) are a job for a separate targeted updater to
            # build later — one that checks specific businesses on purpose,
            # not a side effect of re-walking the same tiles. last_seen_at
            # is still bumped on every re-sighting specifically so that
            # future updater has a real "not re-confirmed since X" signal to
            # work from — if this went to a true no-op entirely,
            # last_seen_at would always equal first_seen_at and that signal
            # would be lost. PyMySQL's executemany rewrites a single
            # VALUES (...) statement into one real multi-row INSERT, the
            # same batching psycopg2.extras.execute_values gave on Postgres.
            # source is likewise only ever set on first insert (ON DUPLICATE
            # KEY UPDATE never touches it) — if Overture and OSM both find
            # the same real-world business, whichever source got there
            # first keeps credit, which is fine since source is "who told
            # us about this row", not "who currently verifies it".
            cur.executemany("""
                INSERT INTO businesses (
                    id, name, category_id, category_extra_id, address, location_id,
                    postcode, phone, website, email, social, brand_id, confidence,
                    status_id, lon, lat, source, basic_category, category_hierarchy, last_seen_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE last_seen_at = VALUES(last_seen_at)
            """, batch)


def load_done_tiles(pg, release):
    """One query for the WHOLE pass instead of one per tile — this is the
    change that keeps tile-done checks off Prisma's metered-operations
    budget almost entirely. Each shard only ever needs ITS OWN done-tiles
    set, never another shard's — tile-to-shard assignment is static for a
    pass (see assign_tiles_to_shards), so a tile this shard hasn't scanned
    yet was never going to be in another shard's scan_tiles either."""
    # `release` is a reserved word in MySQL/TiDB (used by RELEASE SAVEPOINT)
    # — backtick-quoted here and in database/schema.mysql.sql, same as the
    # scan_tiles table's own column definition.
    with pg.cursor() as cur:
        cur.execute("SELECT tile_key FROM scan_tiles WHERE `release` = %s", (release,))
        return {row[0] for row in cur.fetchall()}


def flush_done_tiles(pg, pending):
    """pending: list of (tile_key, release, rows_found) tuples, batched and
    flushed together rather than one insert per tile."""
    if not pending:
        return
    with pg.cursor() as cur:
        cur.executemany("""
            INSERT INTO scan_tiles (tile_key, `release`, rows_found, scanned_at) VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                rows_found = VALUES(rows_found), scanned_at = VALUES(scanned_at)
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


# MySQL/TiDB error codes that mean "the database schema itself doesn't
# match what this code expects" — never transient, never fixed by
# retrying or splitting a region smaller. 1054 = Unknown column, 1146 =
# table doesn't exist, 1049 = unknown database (e.g. a secret still
# pointing at the wrong database name — the exact /sys-vs-company_scrapper
# mistake hit earlier in this project's live deployment).
_SCHEMA_MISMATCH_ERRNOS = {1054, 1146, 1049}


def _is_schema_mismatch_error(e: Exception) -> bool:
    if isinstance(e, pymysql.MySQLError) and e.args:
        try:
            return int(e.args[0]) in _SCHEMA_MISMATCH_ERRNOS
        except (TypeError, ValueError):
            return False
    return False


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
            # Postgres categorically refuses a text value containing a NUL
            # (0x00) byte — not a validation choice, it's a hard limitation
            # of how it stores text internally. Overture's free-text fields
            # (name/address especially) occasionally carry garbage bytes
            # from whatever source scraped them, and a single bad byte in
            # one record otherwise aborts the ENTIRE batch it's bundled
            # into, including however many thousands of good records share
            # that batch — strip it up front rather than let one corrupt
            # record cost the rest of the tile's data.
            records = [
                {k: (v.replace("\x00", "") if isinstance(v, str) else v) for k, v in r.items()}
                for r in records
            ]
            pg_upsert_businesses(pg, lookups, records)
        return n
    except Exception as e:
        # A real incident caught live (Nov 2026): a schema change
        # (adding businesses.basic_category/category_hierarchy) shipped
        # before all live shards had the ALTER TABLE applied — every
        # single tile failed with "Unknown column 'basic_category'", and
        # this function's split-and-retry logic (correctly designed for
        # TRANSIENT failures like an OOM on a dense region, where a
        # smaller bbox genuinely has a better chance) instead spent the
        # entire ~5.5h scheduled run quartering every tile down to its
        # smallest split size and failing identically at every depth —
        # a schema mismatch fails the exact same way no matter how small
        # the region is, so splitting can never help it. Fail the WHOLE
        # run immediately and loudly for this error class instead of
        # burning the scheduled window finding that out one tile at a
        # time.
        if _is_schema_mismatch_error(e):
            raise RuntimeError(
                f"Schema mismatch against this shard's database: {e}. This is NOT a "
                f"transient per-region failure — splitting into smaller regions cannot "
                f"fix a missing/renamed column, so this run is aborting immediately "
                f"instead of burning its whole time budget re-discovering this on every "
                f"tile. Apply the missing schema change (see database/schema.mysql.sql) "
                f"to this shard's database, then re-run."
            ) from e
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


def assign_tiles_to_shards(tiles, num_shards):
    """Round-robins the full tile list across shards by index (tile i goes
    to shard bucket i % num_shards) instead of chopping it into
    contiguous blocks. A contiguous split would hand one shard a solid
    wall of dense-metro tiles (slow, row-heavy) and another shard a solid
    wall of open-ocean tiles (fast, almost empty) — round-robin spreads
    that unevenness across every shard instead, so they all finish around
    the same time and end up with roughly similar row counts."""
    buckets = [[] for _ in range(num_shards)]
    for i, tile in enumerate(tiles):
        buckets[i % num_shards].append(tile)
    return buckets


def run_shard_pass(shard_num, url, tiles, release, con_factory, select, cols, deadline=None):
    """Runs one shard's share of a pass start to finish: its own Postgres
    connection, its own Lookups cache, its own scan_tiles done-set, its
    own DuckDB connection (built fresh via con_factory so threads never
    share a DuckDB connection). Returns True if this shard had any
    not-yet-done tile to work on this call."""
    pg = pg_connect(url)
    con = con_factory()
    try:
        lookups = Lookups(pg)
        done_tiles = load_done_tiles(pg, release)  # ONE query for this shard's whole pass
        pending_done = []
        did_any_work = False

        for i, (x0, y0, x1, y1) in enumerate(tiles, 1):
            if deadline is not None and time.monotonic() >= deadline:
                print(f"[shard {shard_num}] runtime cap reached mid-pass at tile {i}/{len(tiles)} — "
                      f"stopping here, cleanly, nothing lost.")
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
                    print(f"[shard {shard_num}] tile {tile_key} attempt {attempt + 1}/3 failed "
                          f"even after splitting: {e}")
                    time.sleep(5 * (attempt + 1))
            if n is None:
                print(f"[shard {shard_num}] SKIPPED {tile_key} after 3 failed attempts — "
                      f"not marked done, will retry on a later loop iteration.")
                continue

            pending_done.append((tile_key, release, n))
            if len(pending_done) >= 20:
                flush_done_tiles(pg, pending_done)
            print(f"[shard {shard_num}] [{i}/{len(tiles)}] tile {tile_key}: {n:,} US rows synced")

        flush_done_tiles(pg, pending_done)
        return did_any_work
    finally:
        con.close()
        pg.close()


def run_one_pass(release, shards, deadline=None):
    """Builds the tile list and the Overture column-presence probe ONCE,
    splits the tile list across every configured shard, then scans all
    shards CONCURRENTLY — one worker thread per shard, each fully
    independent (own DuckDB connection, own Postgres connection, own
    Lookups cache), so every configured database is being written to at
    the same time rather than one after another."""
    probe = duckdb.connect()
    probe.execute("INSTALL httpfs; LOAD httpfs;")
    probe.execute("SET s3_region='us-west-2';")
    src = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
    probe.execute(f"CREATE VIEW places AS SELECT * FROM read_parquet('{src}', hive_partitioning=1);")
    cols = {r[0] for r in probe.execute("DESCRIBE places").fetchall()}
    probe.close()

    select = f"""
      id,
      {_pick(cols, ("names.primary", "names"))} AS name,
      {_pick(cols, ("taxonomy.primary", "taxonomy"), ("categories.primary", "categories"))} AS category,
      {_pick(cols, ("taxonomy.alternates[1]", "taxonomy"), ("categories.alternate", "categories"))} AS category_extra,
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
      {_pick(cols, ("basic_category", "basic_category"))} AS basic_category,
      {_pick(cols, ("to_json(taxonomy.hierarchy)", "taxonomy"))} AS category_hierarchy,
      bbox.xmin AS lon,
      bbox.ymin AS lat
    """

    tiles = [t for (x0, y0, x1, y1) in US_REGIONS for t in _tiles_for_region(x0, y0, x1, y1, TILE_DEG)]
    buckets = assign_tiles_to_shards(tiles, len(shards))
    summary = ", ".join(f"shard {num}: {len(bucket)} tiles" for (num, _url), bucket in zip(shards, buckets))
    print(f"Pass against release {release}: {len(tiles)} tiles total ({TILE_DEG}x{TILE_DEG} deg each, "
          f"US + SBA territories only), split across {len(shards)} shard database(s) running "
          f"concurrently — {summary}.")

    def con_factory():
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_region='us-west-2';")
        con.execute(f"SET threads={THREADS}; SET memory_limit='{MEMORY_LIMIT}';")
        con.execute(f"CREATE VIEW places AS SELECT * FROM read_parquet('{src}', hive_partitioning=1);")
        return con

    did_any_work = False
    errors = []
    with ThreadPoolExecutor(max_workers=len(shards), thread_name_prefix="shard") as pool:
        futures = {
            pool.submit(run_shard_pass, shard_num, url, bucket, release, con_factory, select, cols, deadline): shard_num
            for (shard_num, url), bucket in zip(shards, buckets)
        }
        for future in as_completed(futures):
            shard_num = futures[future]
            try:
                if future.result():
                    did_any_work = True
            except Exception as e:
                print(f"[shard {shard_num}] pass failed with an unhandled error: {e}")
                errors.append((shard_num, e))
    # every shard's thread has finished (the `with` block above blocks on
    # shutdown until they all complete) before we ever get here, so a
    # failure in one shard never cuts the others' progress short.
    if errors:
        shard_list = ", ".join(str(n) for n, _ in errors)
        raise RuntimeError(f"shard(s) {shard_list} failed this pass — see logs above for each error")
    return did_any_work


def main():
    # MAX_RUNTIME_SECONDS is for a bounded host that re-triggers this
    # process on its own schedule instead of letting it run forever — a
    # GitHub Actions job (6-hour hard cap per job) being the real case this
    # was added for. Unset (the default) means run forever (Render, a VM).
    max_runtime = os.environ.get("MAX_RUNTIME_SECONDS")
    deadline = time.monotonic() + int(max_runtime) if max_runtime else None

    shards = discover_shards()
    shard_nums = ", ".join(str(n) for n, _ in shards)
    print("Company Scrapper scan worker starting — US + SBA territories only, "
          f"no GPU required, no local disk state kept. {len(shards)} shard database(s) configured "
          f"(shard numbers: {shard_nums})."
          + (f" Bounded run: exiting by ~{int(max_runtime)}s from now." if deadline else " Running indefinitely."))

    while True:
        if deadline and time.monotonic() >= deadline:
            print("MAX_RUNTIME_SECONDS reached — exiting cleanly so the scheduler can re-trigger. "
                  "Nothing lost: progress lives in each shard's own scan_tiles table, not in this process.")
            return

        release = latest_release()
        try:
            did_work = run_one_pass(release, shards, deadline)
        except Exception as e:
            print(f"Pass against {release} failed with an unhandled error: {e}. Retrying in 5 minutes.")
            _sleep_respecting_deadline(300, deadline)
            continue

        if not did_work:
            if deadline:
                print(f"Full pass against release {release} already complete across every shard — "
                      f"nothing left to do this run, exiting cleanly.")
                return
            print(f"Full pass against release {release} already complete across every shard. "
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
