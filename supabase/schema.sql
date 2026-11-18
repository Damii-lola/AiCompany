-- Company Scrapper — Supabase schema
-- =====================================================================
-- Run this ONCE in the Supabase SQL editor (Project -> SQL Editor -> New
-- query -> paste -> Run) before the Render service or the Kaggle scanner
-- ever try to write to it. This is the live, continuously-updated store
-- of real businesses the scanner finds — a running table, not a snapshot,
-- which is what makes "constantly updating itself" real: every sighting
-- of a business upserts on its real Overture id, bumping last_seen_at,
-- rather than ever truncating and reloading.

create extension if not exists cube;
create extension if not exists earthdistance;

create table if not exists businesses (
  id              text primary key,         -- Overture Maps place id — the real, stable dedupe key
  name            text,
  category        text,
  category_extra  text,
  address         text,
  city            text,
  region          text,
  postcode        text,
  country         text,
  phone           text,
  website         text,
  email           text,
  social          text,
  brand           text,
  confidence      double precision,
  operating_status text,
  lon             double precision,
  lat             double precision,
  first_seen_at   timestamptz not null default now(),
  last_seen_at    timestamptz not null default now()
);

create index if not exists idx_businesses_category on businesses (category);
create index if not exists idx_businesses_country_region_city on businesses (country, region, city);
create index if not exists idx_businesses_no_website on businesses (id) where (website is null or website = '');
create index if not exists idx_businesses_earth on businesses using gist (ll_to_earth(lat, lon));

-- Radius ("trade area") search — same great-circle math as
-- service_sba_trade_area_snapshot.py's DuckDB version, so a request answered
-- via Supabase and one answered via the Kaggle parquet never disagree.
-- earth_box(...) @> ... uses the GiST index to cheaply rule out rows outside
-- a bounding cube BEFORE the exact earth_distance() math runs on survivors —
-- real query-plan optimization, not just "it happens to work."
create or replace function businesses_within_radius(
  center_lat double precision,
  center_lon double precision,
  radius_miles double precision,
  categories text[] default null,
  only_no_website boolean default false,
  country_filter text default null
) returns setof businesses as $$
  select *
  from businesses
  where lat is not null and lon is not null
    and earth_box(ll_to_earth(center_lat, center_lon), radius_miles * 1609.34) @> ll_to_earth(lat, lon)
    and earth_distance(ll_to_earth(center_lat, center_lon), ll_to_earth(lat, lon)) <= radius_miles * 1609.34
    and (categories is null or category = any(categories))
    and (only_no_website = false or website is null or trim(website) = '')
    and (country_filter is null or country = country_filter)
  order by earth_distance(ll_to_earth(center_lat, center_lon), ll_to_earth(lat, lon)) asc;
$$ language sql stable;

-- Trade-area summary (competitor_count + no_website_count) in ONE round trip
-- instead of two — this is the exact pair of numbers the locked SBA V1
-- product needs, computed server-side rather than pulled client-side and
-- counted in JS.
create or replace function businesses_trade_area_summary(
  center_lat double precision,
  center_lon double precision,
  radius_miles double precision,
  categories text[] default null,
  country_filter text default null
) returns table(competitor_count bigint, no_website_count bigint, nearest_miles double precision) as $$
  select
    count(*) as competitor_count,
    count(*) filter (where website is null or trim(website) = '') as no_website_count,
    min(earth_distance(ll_to_earth(center_lat, center_lon), ll_to_earth(lat, lon)) / 1609.34) as nearest_miles
  from businesses
  where lat is not null and lon is not null
    and earth_box(ll_to_earth(center_lat, center_lon), radius_miles * 1609.34) @> ll_to_earth(lat, lon)
    and earth_distance(ll_to_earth(center_lat, center_lon), ll_to_earth(lat, lon)) <= radius_miles * 1609.34
    and (categories is null or category = any(categories))
    and (country_filter is null or country = country_filter);
$$ language sql stable;

-- Scan progress — lets the scanner (now a Render background worker, not a
-- Kaggle notebook) be fully stateless on local disk. tile_key is scoped by
-- release ("2026-10-21.0:-180_-90") so a newer Overture release's tiles are
-- automatically "not done" with no reset/clear step ever needed — old
-- releases' rows just sit here harmlessly (a few thousand tiny rows per
-- release, trivial for Postgres).
create table if not exists scan_tiles (
  tile_key    text primary key,
  release     text not null,
  rows_found  integer not null default 0,
  scanned_at  timestamptz not null default now()
);
create index if not exists idx_scan_tiles_release on scan_tiles (release);
