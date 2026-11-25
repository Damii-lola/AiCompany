-- Company Scrapper — TiDB Cloud Starter schema (MySQL dialect)
-- =====================================================================
-- Replaces database/schema.sql (Postgres) for the move off Prisma
-- Postgres, forced by Prisma's 2GB/workspace free-tier cap and the
-- unavailability of enough separate emails to keep adding Prisma
-- accounts. TiDB Cloud Starter gives up to 5 free instances PER
-- ACCOUNT (no card needed for those 5), 5GiB storage + 50M request
-- units/month each — 25GB total under one login.
--
-- Run this ONCE against each TiDB Cloud Starter instance (mysql
-- "$DATABASE_URL" < database/schema.mysql.sql, or paste into TiDB
-- Cloud's SQL console) before render-service or the scanner write to
-- it. Verified against a real local MariaDB 10.11 instance in this
-- session (MariaDB is MySQL-wire-compatible, used as a syntax/logic
-- stand-in) — final confirmation still needed against a real TiDB
-- Cloud Starter instance, since TiDB has its own engine-level quirks
-- MariaDB won't catch.
--
-- WHAT CHANGED FROM THE POSTGRES VERSION, AND WHY:
--   - No `create extension cube/earthdistance` — TiDB has neither.
--     Radius ("trade area") search moved OUT of SQL entirely: this
--     schema only adds a plain (lat, lon) index for a cheap bounding-
--     box pre-filter; the exact great-circle distance is computed in
--     JS in render-service/lib/trade-area.js, the same place that
--     already merges per-shard results together.
--   - No `businesses_within_radius`/`businesses_trade_area_summary`
--     SQL functions — TiDB doesn't support stored functions/
--     procedures at all. Same application-code replacement as above.
--   - No GiST index — replaced by a plain btree-style composite index
--     on (lat, lon) for the bounding-box pre-filter.
--   - No per-column LZ4 TOAST compression syntax (Postgres-specific) —
--     TiDB's storage engine (TiKV) compresses data at the storage
--     layer automatically; there's no per-column DDL for it, and
--     nothing is lost by dropping the ALTER statements.
--   - smallserial/serial -> *_unsigned auto_increment, text primary
--     keys/foreign-key targets -> varchar(n) (MySQL/TiDB can't index
--     or primary-key an unbounded TEXT column without a prefix
--     length, so every column that's ever filtered, joined, or
--     sorted on gets a concrete varchar length instead).
--   - `create index if not exists` keeps the exact same "always safe
--     to re-run" property the Postgres version had.
--   - The no-website partial index became a real equivalent: TiDB
--     supports generated columns, so has_website is computed and
--     indexed instead of indexing a partial WHERE condition (which
--     MySQL/TiDB don't support).
-- =====================================================================

create table if not exists categories (
  id   smallint unsigned auto_increment primary key,
  name varchar(255) not null,
  unique key uniq_categories_name (name)
);

-- city/region default to '' rather than null, same reasoning as the
-- Postgres version: a unique constraint treats every null as distinct,
-- which would let duplicate (null, null, 'US') rows pile up. '' is
-- normalized back to null in businesses_readable below.
create table if not exists locations (
  id      int unsigned auto_increment primary key,
  city    varchar(255) not null default '',
  region  varchar(255) not null default '',
  country varchar(255) not null,
  unique key uniq_locations_city_region_country (city, region, country)
);

create table if not exists brands (
  id   int unsigned auto_increment primary key,
  name varchar(255) not null,
  unique key uniq_brands_name (name)
);

create table if not exists statuses (
  id   smallint unsigned auto_increment primary key,
  name varchar(64) not null,
  unique key uniq_statuses_name (name)
);

-- ---------------------------------------------------------------------
-- businesses — same real Overture id as the primary key / dedupe key.
-- varchar(64) comfortably covers Overture's GERS id format.
-- ---------------------------------------------------------------------
create table if not exists businesses (
  id                varchar(64) primary key,
  name              varchar(255),
  category_id       smallint unsigned,
  category_extra_id smallint unsigned,
  -- address/phone/website/email/social are never filtered, sorted, or
  -- joined on directly (only the generated+indexed has_website column
  -- below is) — so, matching the original Postgres schema's unbounded
  -- `text` for these, they're `text` here too, not a guessed varchar(n).
  -- A prior varchar(500) cap on website caused real DataError 1406s
  -- ("Data too long") against real Overture data (long tracking-
  -- parameter URLs genuinely exceed 500 chars) — this removes that
  -- ceiling entirely rather than guessing a bigger number.
  address           text,
  location_id       int unsigned,
  -- postcode was assumed short (varchar(20)) matching the original
  -- Postgres schema's comment ("already short, 5-10 bytes") — real
  -- Overture data hit DataError 1406 against that cap too (some
  -- postcode values are longer/combined). Never filtered, sorted, or
  -- indexed anywhere (grepped render-service to confirm), so widened to
  -- TEXT like the other free-text columns rather than guessing a new cap.
  postcode          text,
  phone             text,
  website           text,
  email             text,
  social            text,
  brand_id          int unsigned,
  confidence        float,
  status_id         smallint unsigned,
  lon               float,
  lat               float,
  -- Which discovery source first added this row — 'overture' (default,
  -- the original backfill), 'osm' (OpenStreetMap Overpass, new), or
  -- 'google_maps' (the self-hosted scraper, future). Doubles as the raw
  -- "Data source" field the SBA report spec's section 12 asks for.
  source            varchar(32) not null default 'overture',
  -- Real structural category data straight from Overture's data, for the
  -- on-demand competitive-classification enrichment tier (see
  -- service_competitive_classifier.py). basic_category is Overture's own
  -- single "cognitively natural" grouping label (e.g. 'casual_eatery');
  -- category_hierarchy is the FULL general->specific taxonomy path as a
  -- JSON array string (e.g. '["food_and_drink","restaurant",
  -- "casual_eatery","gas_station_sushi"]'), straight from Overture's
  -- `taxonomy.hierarchy` field (confirmed real, current Overture Places
  -- schema field — docs.overturemaps.org/schema/reference/places/types/
  -- taxonomy/). TEXT, not a fixed-length type: a JSON array has no fixed
  -- byte cap, same reasoning as website/address above. Both NULL for any
  -- row scanned before this column existed, and for OSM rows (OSM tags
  -- have no equivalent taxonomy concept) — the classifier is built to
  -- fall back honestly to leaf-category-only comparison when these are
  -- absent, never to guess a hierarchy that was never actually scanned.
  basic_category    varchar(128),
  category_hierarchy text,
  -- VIRTUAL, not STORED: TiDB does not support adding a STORED generated
  -- column via ALTER TABLE at all (only at CREATE TABLE time), which a
  -- real live shard hit when widening `website` to TEXT required
  -- dropping and re-adding this column. VIRTUAL has no such restriction,
  -- is still indexable in TiDB, and the filter (has_website = 0) uses the
  -- index either way — verified against real local data.
  has_website       boolean as (website is not null and website <> '') virtual,
  first_seen_at     timestamp not null default current_timestamp,
  last_seen_at      timestamp not null default current_timestamp,
  constraint fk_businesses_category       foreign key (category_id)       references categories(id),
  constraint fk_businesses_category_extra foreign key (category_extra_id) references categories(id),
  constraint fk_businesses_location       foreign key (location_id)       references locations(id),
  constraint fk_businesses_brand          foreign key (brand_id)          references brands(id),
  constraint fk_businesses_status         foreign key (status_id)         references statuses(id)
);

create index if not exists idx_businesses_category on businesses (category_id);
create index if not exists idx_businesses_location on businesses (location_id);
create index if not exists idx_businesses_status on businesses (status_id);
create index if not exists idx_businesses_has_website on businesses (has_website);
create index if not exists idx_businesses_name on businesses (name);
create index if not exists idx_businesses_category_name on businesses (category_id, name);
create index if not exists idx_businesses_location_name on businesses (location_id, name);
-- Bounding-box pre-filter index for radius search (see top-of-file
-- note) — replaces the Postgres GiST earthdistance index.
create index if not exists idx_businesses_latlon on businesses (lat, lon);

-- ---------------------------------------------------------------------
-- Convenience view — same flat, human-readable shape as the Postgres
-- version. MySQL/TiDB views are read-only (no INSERT/UPDATE/DELETE
-- through them), which is all render-service ever did with this one.
-- ---------------------------------------------------------------------
create or replace view businesses_readable as
  select
    b.id, b.name,
    c1.name as category,
    c2.name as category_extra,
    b.address,
    nullif(l.city, '') as city, nullif(l.region, '') as region, l.country,
    b.postcode, b.phone, b.website, b.email, b.social,
    br.name as brand,
    b.confidence,
    s.name as operating_status,
    b.lon, b.lat,
    b.has_website,
    b.source,
    b.basic_category, b.category_hierarchy,
    b.first_seen_at, b.last_seen_at
  from businesses b
  left join categories c1 on c1.id = b.category_id
  left join categories c2 on c2.id = b.category_extra_id
  left join locations  l  on l.id  = b.location_id
  left join brands     br on br.id = b.brand_id
  left join statuses   s  on s.id  = b.status_id;

-- ---------------------------------------------------------------------
-- Scan progress — unchanged in shape from the Postgres version.
-- ---------------------------------------------------------------------
-- `release` is a reserved word in MySQL/TiDB (used by RELEASE SAVEPOINT),
-- so it's backtick-quoted everywhere it's used as a column name here and
-- in scanner-service/scan_worker.py's SQL.
create table if not exists scan_tiles (
  tile_key    varchar(64) primary key,
  `release`   varchar(64) not null,
  rows_found  int not null default 0,
  scanned_at  timestamp not null default current_timestamp
);
create index if not exists idx_scan_tiles_release on scan_tiles (`release`);
