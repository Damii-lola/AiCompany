# %% [markdown]
# # Business Database Builder — ONE file on disk, always
# No tile files, ever. Scan progress lives inside a single DuckDB file
# (/kaggle/working/business_scan.duckdb) which doubles as the resume checkpoint.
# At the end, that single file is exported to ONE heavily-compressed .parquet
# and pushed to your Kaggle Dataset. Settings: Internet ON, Secrets: KAGGLE_USERNAME/KAGGLE_KEY attached.

# %% Install
!pip install -q duckdb requests pandas pyarrow kaggle

# %% Config
import os, json, time, glob, shutil, requests, duckdb, pandas as pd

DB_FILE = "/kaggle/working/business_scan.duckdb"     # <- the ONE working file (data + progress)
TMP_TILE = "/kaggle/working/_tmp_tile.parquet"       # reused every loop, never accumulates
FINAL_FILE = "/kaggle/working/business_database.parquet"   # <- the ONE final deliverable
TILE_DEG = 15
MIN_CONFIDENCE = 0.0
THREADS = 4
MEMORY_LIMIT = "12GB"
DATASET_SLUG = "global-businesses-v1"
DATASET_TITLE = "Global Businesses v1"
CHECKPOINT_SLUG = "global-businesses-checkpoint"
CHECKPOINT_TITLE = "Global Businesses Checkpoint (in progress)"
CHECKPOINT_EVERY = 20
COMPRESSION_LEVEL = 22   # zstd "ultra" — smallest file, slower to write once at the end

# Re-running cells in the SAME kernel session (instead of restarting it) leaves last
# run's DuckDB connection open in memory. If so, reconnecting to the same path reuses
# that stale handle instead of a real fresh file — so close anything left over first.
try:
    con.close()
    del con
    print("Closed a leftover connection from a previous run in this session.")
except NameError:
    pass

# %% Kaggle API auth
from kaggle_secrets import UserSecretsClient
_s = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = _s.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = _s.get_secret("KAGGLE_KEY")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
USER = os.environ["KAGGLE_USERNAME"]

def _exists_error(e):
    s = str(e).lower()
    return "already in use" in s or "already exists" in s

def push_dataset(folder, slug, title, note):
    """Try to create the dataset. If it already exists, push a new version instead.
    CONFIRMED BUG (seen live, elsewhere in this project, same pattern): this used
    to return False and let the caller silently continue on a real write failure
    — checkpoint calls during scan_world() don't check the return value, so a
    broken write could burn hours of scan time before the FINAL push (the only
    one that was ever checked) finally surfaced it. Now it raises immediately
    instead, so a dead write path is caught at the very next checkpoint, not
    hours later."""
    json.dump({"title": title, "id": f"{USER}/{slug}", "licenses": [{"name": "CC0-1.0"}]},
              open(f"{folder}/dataset-metadata.json", "w"))
    try:
        api.dataset_create_new(folder, public=False, dir_mode="skip")
        print(f"  -> created dataset {slug}")
        return True
    except Exception as e:
        if _exists_error(e):
            try:
                api.dataset_create_version(folder, version_notes=note, dir_mode="skip")
                print(f"  -> pushed new version to {slug}")
                return True
            except Exception as e2:
                raise RuntimeError(f"Push failed (version) for {slug}: {e2}") from e2
        raise RuntimeError(f"Push failed (create) for {slug}: {e}") from e

def push_single_file(local_path, slug, title, note):
    """Hard-links local_path into its own upload folder (so only ONE file is ever
    sent) and pushes it. No extra disk space used."""
    up = f"/kaggle/working/_up_{slug}"
    os.makedirs(up, exist_ok=True)
    for f in glob.glob(f"{up}/*"):
        os.remove(f)
    link = os.path.join(up, os.path.basename(local_path))
    os.link(local_path, link)
    return push_dataset(up, slug, title, note)

# %% Real-time ingest into Supabase, via the Render service — the "constantly
# updating itself" step. Supabase credentials never touch this notebook at
# all (only Render holds them, per the architecture decision); this script
# only ever knows a shared secret and the Render URL, both Kaggle Secrets.
# Deliberately NON-FATAL: a Render/network hiccup here must never kill a
# multi-hour world scan. The Kaggle Dataset checkpoint above (and the final
# parquet export below) stay the real source of truth either way — this is
# a live mirror into Supabase, not a replacement for them.
try:
    RENDER_INGEST_URL = _s.get_secret("RENDER_INGEST_URL").rstrip("/") + "/api/ingest/businesses"
    INGEST_SECRET = _s.get_secret("INGEST_SECRET")
    INGEST_ENABLED = True
    print(f"Live Supabase ingest ENABLED -> {RENDER_INGEST_URL}")
except Exception as e:
    INGEST_ENABLED = False
    print(f"Live Supabase ingest DISABLED (no RENDER_INGEST_URL/INGEST_SECRET secret set: {e}). "
          f"Scan will still produce the Kaggle checkpoint/final parquet as always — this is optional.")

INGEST_BATCH_SIZE = 500   # keep each POST small; a 15-degree tile can have many thousands of rows


def ingest_tile_to_render(df):
    """Pushes one tile's real rows to the Render service, chunked, retried
    once, and never allowed to raise past this function — see the module
    docstring above for why a failure here must stay non-fatal."""
    if not INGEST_ENABLED or df.empty:
        return
    # pandas -> JSON-safe records: to_json turns NaN into real null, which
    # json.loads then gives us back as None — plain json.dumps on a raw
    # pandas record dict would choke on NaN instead.
    records = json.loads(df.to_json(orient="records"))
    sent, failed = 0, 0
    for i in range(0, len(records), INGEST_BATCH_SIZE):
        batch = records[i:i + INGEST_BATCH_SIZE]
        for attempt in range(2):
            try:
                r = requests.post(
                    RENDER_INGEST_URL,
                    headers={"x-ingest-secret": INGEST_SECRET, "Content-Type": "application/json"},
                    json={"businesses": batch}, timeout=30,
                )
                if r.status_code == 200:
                    sent += len(batch)
                else:
                    print(f"  ingest batch failed (HTTP {r.status_code}): {r.text[:200]}")
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(3)
                    continue
                print(f"  ingest batch failed after retry: {e}")
                failed += len(batch)
    if failed:
        print(f"  ingest: {sent} rows synced to Supabase, {failed} rows failed this tile "
              f"(non-fatal — they're still safe in the Kaggle checkpoint, will retry next run's tiles)")

# %% Always pull from the checkpoint dataset — the ONLY source of truth.
# Local disk is never trusted, never checked, and treated as fully disposable:
# anything already sitting in /kaggle/working is overwritten by what's on Kaggle.
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)   # local copy means nothing; the dataset is ground truth

try:
    restore_dir = "/kaggle/working/_restore"
    if os.path.exists(restore_dir):
        shutil.rmtree(restore_dir)
    os.makedirs(restore_dir, exist_ok=True)
    api.dataset_download_files(f"{USER}/{CHECKPOINT_SLUG}", path=restore_dir, unzip=True, quiet=False)
    found = glob.glob(f"{restore_dir}/**/*.duckdb", recursive=True)
    if found:
        shutil.move(found[0], DB_FILE)
        print(f"Pulled checkpoint from the Kaggle Dataset: {DB_FILE}")
    else:
        print("Checkpoint dataset exists but had no .duckdb file in it — this is a true first run.")
except Exception as e:
    print(f"No checkpoint dataset yet on Kaggle — this is a true first run. ({e})")

# %% Find latest Overture release
def latest_release():
    try:
        r = requests.get("https://stac.overturemaps.org/catalog.json", timeout=30).json()
        return r["latest"]
    except Exception as e:
        print("STAC lookup failed, using fallback:", e)
        return "2026-10-21.0"

RELEASE = latest_release()
SRC = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"
print("Release:", RELEASE)

# %% Connect to the single DuckDB file (this IS the database — no separate state.json)
con = duckdb.connect(DB_FILE)
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_region='us-west-2';")
con.execute(f"SET threads={THREADS}; SET memory_limit='{MEMORY_LIMIT}';")
con.execute(f"CREATE VIEW IF NOT EXISTS places AS SELECT * FROM read_parquet('{SRC}', hive_partitioning=1);")
cols = {r[0] for r in con.execute("DESCRIBE places").fetchall()}

def pick(*options):
    for expr, needed in options:
        if needed in cols:
            return expr
    return "NULL"

SELECT = f"""
  id,
  {pick(("names.primary","names"))} AS name,
  {pick(("taxonomy.primary","taxonomy"),("categories.primary","categories"))} AS category,
  {pick(("taxonomy.hierarchy","taxonomy"),("categories.alternate","categories"))} AS category_extra,
  {pick(("addresses[1].freeform","addresses"))} AS address,
  {pick(("addresses[1].locality","addresses"))} AS city,
  {pick(("addresses[1].region","addresses"))} AS region,
  {pick(("addresses[1].postcode","addresses"))} AS postcode,
  {pick(("addresses[1].country","addresses"))} AS country,
  {pick(("phones[1]","phones"))} AS phone,
  {pick(("websites[1]","websites"))} AS website,
  {pick(("emails[1]","emails"))} AS email,
  {pick(("socials[1]","socials"))} AS social,
  {pick(("brand.names.primary","brand"))} AS brand,
  {pick(("confidence","confidence"))} AS confidence,
  {pick(("operating_status","operating_status"))} AS operating_status,
  bbox.xmin AS lon,
  bbox.ymin AS lat
"""

# %% Tables inside the single DB file: the data itself, and a tiny progress log
con.execute(f"CREATE TABLE IF NOT EXISTS businesses AS SELECT {SELECT} FROM places LIMIT 0")
con.execute("CREATE TABLE IF NOT EXISTS scan_progress (tile_key VARCHAR PRIMARY KEY, rows BIGINT)")

def tile_done(key):
    return con.execute("SELECT 1 FROM scan_progress WHERE tile_key = ?", [key]).fetchone() is not None

def mark_done(key, n):
    con.execute("INSERT OR REPLACE INTO scan_progress VALUES (?, ?)", [key, n])

def total_rows():
    return con.execute("SELECT count(*) FROM businesses").fetchone()[0]

# %% Scan the world, inserting straight into the one table (no tile files kept)
def scan_world():
    tiles = [(x, y) for x in range(-180, 180, TILE_DEG) for y in range(-90, 90, TILE_DEG)]
    t0 = time.time()
    since_checkpoint = 0
    for i, (x, y) in enumerate(tiles, 1):
        key = f"{x}_{y}"
        if tile_done(key):
            continue
        where = (f"bbox.xmin >= {x} AND bbox.xmin < {x+TILE_DEG} "
                 f"AND bbox.ymin >= {y} AND bbox.ymin < {y+TILE_DEG}")
        if "confidence" in cols and MIN_CONFIDENCE > 0:
            where += f" AND confidence >= {MIN_CONFIDENCE}"

        for attempt in range(4):
            try:
                if os.path.exists(TMP_TILE):
                    os.remove(TMP_TILE)
                con.execute(f"COPY (SELECT {SELECT} FROM places WHERE {where}) "
                            f"TO '{TMP_TILE}' (FORMAT PARQUET, COMPRESSION ZSTD)")
                break
            except Exception as e:
                print(f"tile {key} attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            print(f"SKIPPED {key} after retries; re-run to retry it")
            continue

        n = con.execute(f"SELECT count(*) FROM read_parquet('{TMP_TILE}')").fetchone()[0]
        if n > 0:
            con.execute(f"INSERT INTO businesses SELECT * FROM read_parquet('{TMP_TILE}')")
            if INGEST_ENABLED:
                ingest_tile_to_render(con.execute(f"SELECT * FROM read_parquet('{TMP_TILE}')").fetchdf())
        os.remove(TMP_TILE)   # the only tile-shaped file is gone again immediately
        mark_done(key, n)
        since_checkpoint += 1
        print(f"[{i}/{len(tiles)}] tile {key}: {n:,} | total {total_rows():,} | {time.time()-t0:.0f}s")

        if since_checkpoint >= CHECKPOINT_EVERY:
            print(f"Checkpointing ({total_rows():,} rows so far)...")
            con.execute("CHECKPOINT")   # flush DuckDB's WAL into the single file on disk
            push_single_file(DB_FILE, CHECKPOINT_SLUG, CHECKPOINT_TITLE, f"{total_rows():,} rows, tile {i}/{len(tiles)}")
            since_checkpoint = 0

    print("SCAN DONE. Total businesses:", f"{total_rows():,}")
    con.execute("CHECKPOINT")
    push_single_file(DB_FILE, CHECKPOINT_SLUG, CHECKPOINT_TITLE, f"scan complete, {total_rows():,} rows")

scan_world()

# %% Export the single table to ONE heavily-compressed final file.
# Row groups are set large on purpose: Parquet dictionary-encodes repeated values
# (country codes, category names) per row group, so a bigger row group means one
# shared dictionary covers far more rows instead of being rebuilt every group —
# this alone typically shrinks low-cardinality columns far more than compression level does.
ROW_GROUP_SIZE = 2_000_000

def try_export(level, row_group_size):
    con.execute(f"""
      COPY businesses TO '{FINAL_FILE}'
      (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL {level}, ROW_GROUP_SIZE {row_group_size})
    """)

export_attempts = [
    (22, ROW_GROUP_SIZE),   # zstd "ultra" + large row groups — best compression, slowest
    (19, ROW_GROUP_SIZE),   # fallback if 22 isn't supported by this duckdb build
    (19, None),             # fallback if ROW_GROUP_SIZE option isn't supported
]
exported = False
for level, rgs in export_attempts:
    try:
        if rgs:
            try_export(level, rgs)
        else:
            con.execute(f"COPY businesses TO '{FINAL_FILE}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL {level})")
        print(f"Exported with COMPRESSION_LEVEL {level}" + (f", ROW_GROUP_SIZE {rgs}" if rgs else "") + ".")
        exported = True
        break
    except Exception as e:
        print(f"Export with level={level}, row_group_size={rgs} failed ({e}), trying next option...")

if not exported:
    con.execute(f"COPY businesses TO '{FINAL_FILE}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print("Exported with default ZSTD settings (all higher-compression options unsupported here).")

final_rows = con.execute(f"SELECT count(*) FROM read_parquet('{FINAL_FILE}')").fetchone()[0]
db_rows = total_rows()
assert final_rows == db_rows, f"Mismatch: exported={final_rows} vs scanned={db_rows}"
print(f"Verified: {final_rows:,} rows, {os.path.getsize(FINAL_FILE)/1024**3:.2f} GB (single file)")

# %% Sanity check
db = duckdb.connect()
print(db.execute(f"SELECT count(*) AS total FROM read_parquet('{FINAL_FILE}')").df())
print(db.execute(f"SELECT country, count(*) n FROM read_parquet('{FINAL_FILE}') GROUP BY 1 ORDER BY n DESC LIMIT 10").df())

# %% Push the single final file to its own Kaggle Dataset
ok = push_single_file(FINAL_FILE, DATASET_SLUG, DATASET_TITLE, f"{final_rows:,} businesses, single compressed file")
if ok:
    print(f"\nSaved. Use in future notebooks as: /kaggle/input/{DATASET_SLUG}/{os.path.basename(FINAL_FILE)}")
    print("You can now delete the checkpoint dataset from kaggle.com — it's no longer needed.")
else:
    print(f"\nFinal push failed, but the file is safe at {FINAL_FILE}. Re-run this cell to retry.")

# %% Close the connection so a later re-run in this SAME session starts genuinely
# fresh instead of silently reusing this run's open database.
con.close()
del con
