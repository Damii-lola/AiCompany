-- Company Scrapper — Prisma Postgres schema (replaces supabase/schema.sql)
-- =====================================================================
-- Run this ONCE against your Prisma Postgres database (psql "$DATABASE_URL"
-- -f database/schema.sql, or paste into Prisma's SQL console) before the
-- Render service or the scanner ever try to write to it.
--
-- Why this looks different from the old Supabase schema: that one stored
-- every value (category, city, region, country, brand, status) as a raw
-- text string on every single row. At US-wide scale that's millions of
-- rows each repeating the same ~20 distinct categories, ~60 states/
-- territories, ~30k cities and a few thousand brand names as full text,
-- over and over. This version normalizes those low-cardinality, high-
-- repetition columns into small lookup tables and stores a 2-4 byte
-- integer reference instead — the single biggest lever for fitting inside
-- Prisma Postgres's free-tier storage. The remaining free-text columns
-- (name, address, phone, website, email, social) get LZ4 TOAST
-- compression, which costs nothing to query and shrinks storage for any
-- value long enough to be worth compressing.
--
-- cube/earthdistance are confirmed supported on Prisma Postgres, so the
-- radius-search math is unchanged from the Supabase version.

create extension if not exists cube;
create extension if not exists earthdistance;

-- ---------------------------------------------------------------------
-- Lookup tables — small, loaded into memory once per scan run instead of
-- being re-queried per row (keeps us well under Prisma's metered
-- operations/month).
-- ---------------------------------------------------------------------
create table if not exists categories (
  id   smallserial primary key,
  name text not null unique
);

-- city/region default to '' rather than null: Postgres treats every null
-- as distinct for a unique constraint, which would let duplicate
-- (null, null, 'US') rows pile up. '' is normalized back to null in the
-- businesses_readable view below, so callers never see the difference.
create table if not exists locations (
  id      serial primary key,
  city    text not null default '',
  region  text not null default '',
  country text not null,
  unique (city, region, country)
);

create table if not exists brands (
  id   serial primary key,
  name text not null unique
);

create table if not exists statuses (
  id   smallserial primary key,
  name text not null unique
);

-- ---------------------------------------------------------------------
-- businesses — same real Overture id as the primary key / dedupe key as
-- before. category_id/category_extra_id both point at the one shared
-- categories table (same domain of values). postcode stays inline text —
-- it's already short (5-10 bytes) and too high-cardinality to normalize
-- usefully against ~30k distinct US zip codes.
-- ---------------------------------------------------------------------
create table if not exists businesses (
  id                text primary key,
  name              text,
  category_id       smallint references categories(id),
  category_extra_id smallint references categories(id),
  address           text,
  location_id       integer references locations(id),
  postcode          text,
  phone             text,
  website           text,
  email             text,
  social            text,
  brand_id          integer references brands(id),
  confidence        real,              -- was double precision (8 bytes) -> real (4 bytes)
  status_id         smallint references statuses(id),
  lon               real,              -- was double precision -> real; ~1cm precision loss, irrelevant at business-address scale
  lat               real,
  first_seen_at     timestamptz not null default now(),
  last_seen_at      timestamptz not null default now()
);

-- LZ4 TOAST compression on the remaining free-text columns (Postgres 14+,
-- confirmed available on Prisma Postgres). Free to apply, only affects
-- values long enough to be TOASTed in the first place.
alter table businesses alter column name    set compression lz4;
alter table businesses alter column address set compression lz4;
alter table businesses alter column website set compression lz4;
alter table businesses alter column email   set compression lz4;
alter table businesses alter column social  set compression lz4;

create index if not exists idx_businesses_category on businesses (category_id);
create index if not exists idx_businesses_location on businesses (location_id);
create index if not exists idx_businesses_no_website on businesses (id) where (website is null or website = '');
create index if not exists idx_businesses_earth on businesses using gist (ll_to_earth(lat, lon));

-- ---------------------------------------------------------------------
-- Convenience view — gives back the same flat, human-readable shape the
-- old Supabase table had (real text, not ids). Render's API reads through
-- this so its query code barely changes, and the storage win is entirely
-- in the underlying table.
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
    b.first_seen_at, b.last_seen_at
  from businesses b
  left join categories c1 on c1.id = b.category_id
  left join categories c2 on c2.id = b.category_extra_id
  left join locations  l  on l.id  = b.location_id
  left join brands     br on br.id = b.brand_id
  left join statuses   s  on s.id  = b.status_id;

-- ---------------------------------------------------------------------
-- Radius ("trade area") search — same great-circle math as before, now
-- joining out to categories/locations instead of comparing raw text.
-- category names in -> resolved to category_id internally so callers
-- never have to know ids exist.
-- ---------------------------------------------------------------------
-- returns a real TABLE (not "setof businesses_readable") specifically so
-- distance_miles comes back as a genuine, queryable output column rather
-- than something only this function's internal ORDER BY knows about.
-- That matters once results from several shard databases need to be
-- merged in application code (render-service/lib/db.js) and re-sorted
-- by actual distance across all of them together — ordering each
-- shard's own rows internally isn't enough once there's more than one
-- shard's results to interleave.
create or replace function businesses_within_radius(
  center_lat double precision,
  center_lon double precision,
  radius_miles double precision,
  categories_in text[] default null,
  only_no_website boolean default false,
  country_filter text default null
) returns table (
  id text, name text, category text, category_extra text, address text,
  city text, region text, country text, postcode text, phone text,
  website text, email text, social text, brand text, confidence real,
  operating_status text, lon real, lat real,
  first_seen_at timestamptz, last_seen_at timestamptz,
  distance_miles double precision
) as $$
  select r.*,
    earth_distance(ll_to_earth(center_lat, center_lon), ll_to_earth(r.lat, r.lon)) / 1609.34 as distance_miles
  from businesses_readable r
  where r.lat is not null and r.lon is not null
    and earth_box(ll_to_earth(center_lat, center_lon), radius_miles * 1609.34) @> ll_to_earth(r.lat, r.lon)
    and earth_distance(ll_to_earth(center_lat, center_lon), ll_to_earth(r.lat, r.lon)) <= radius_miles * 1609.34
    and (categories_in is null or r.category = any(categories_in))
    and (only_no_website = false or r.website is null or trim(r.website) = '')
    and (country_filter is null or r.country = country_filter)
  order by distance_miles asc;
$$ language sql stable;

create or replace function businesses_trade_area_summary(
  center_lat double precision,
  center_lon double precision,
  radius_miles double precision,
  categories_in text[] default null,
  country_filter text default null
) returns table(competitor_count bigint, no_website_count bigint, nearest_miles double precision) as $$
  select
    count(*) as competitor_count,
    count(*) filter (where r.website is null or trim(r.website) = '') as no_website_count,
    min(earth_distance(ll_to_earth(center_lat, center_lon), ll_to_earth(r.lat, r.lon)) / 1609.34) as nearest_miles
  from businesses_readable r
  where r.lat is not null and r.lon is not null
    and earth_box(ll_to_earth(center_lat, center_lon), radius_miles * 1609.34) @> ll_to_earth(r.lat, r.lon)
    and earth_distance(ll_to_earth(center_lat, center_lon), ll_to_earth(r.lat, r.lon)) <= radius_miles * 1609.34
    and (categories_in is null or r.category = any(categories_in))
    and (country_filter is null or r.country = country_filter);
$$ language sql stable;

-- ---------------------------------------------------------------------
-- Scan progress — unchanged from the Supabase version, tiny table either way.
-- ---------------------------------------------------------------------
create table if not exists scan_tiles (
  tile_key    text primary key,
  release     text not null,
  rows_found  integer not null default 0,
  scanned_at  timestamptz not null default now()
);
create index if not exists idx_scan_tiles_release on scan_tiles (release);
