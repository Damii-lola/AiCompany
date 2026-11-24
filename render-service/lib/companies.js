"use strict";
// Backs the "Find companies" browse UI (Apollo-style: pick real filters
// from checkboxes, see a paginated real result table) — the user's
// explicit replacement for free-text search as the PRIMARY way to find
// companies ("the way we were doing it before... let's do it apollo
// way... checkboxes to find it"). /api/chat (free-text, with the trained
// understanding worker) stays available as a secondary feature; this is
// the deterministic, always-available, zero-ambiguity path.
//
// Every filter option a user can pick comes from lib/filters.js /
// lib/categories.js's live queries against the real database — nothing
// here is a hardcoded guess at what categories/regions/statuses exist.

const { queryEachShard } = require("./db");

// Short-lived response cache keyed by the exact filter+page combination.
// The single most-repeated request this service ever sees is the
// filter-free, page-1 load every visitor's browser fires the instant the
// page opens — before this cache existed, THAT exact request ran a real
// count(*) and a real ORDER BY/LIMIT scan against all 4 shards' full
// multi-million-row businesses_readable view, from scratch, for every
// single visitor, even two visitors loading the page the same second.
// The real data behind it only changes on the order of hours
// (scan_worker.py runs on a 6-hour cron), so serving a few-seconds-old
// real answer instead of re-querying is still a true, current answer —
// never a fabricated one — just not wastefully re-fetched. Capped so a
// flood of distinct filter combinations can't grow this unbounded.
const RESPONSE_CACHE_TTL_MS = 30 * 1000;
const RESPONSE_CACHE_MAX_ENTRIES = 500;
const responseCache = new Map(); // key -> { result, cachedAt }

function cacheKey(filters, pageNum) {
  return JSON.stringify([
    [...(filters.categories || [])].sort(),
    [...(filters.countries || [])].sort(),
    [...(filters.regions || [])].sort(),
    [...(filters.statuses || [])].sort(),
    (filters.city || "").trim().toLowerCase(),
    !!filters.noWebsiteOnly,
    pageNum,
  ]);
}

function getCached(key) {
  const hit = responseCache.get(key);
  if (!hit) return null;
  if (Date.now() - hit.cachedAt > RESPONSE_CACHE_TTL_MS) {
    responseCache.delete(key);
    return null;
  }
  return hit.result;
}

function setCached(key, result) {
  if (responseCache.size >= RESPONSE_CACHE_MAX_ENTRIES) {
    // Evict the oldest entry rather than letting an unbounded flood of
    // distinct filter combos (or random query-string spam) grow this
    // map forever.
    const oldestKey = responseCache.keys().next().value;
    responseCache.delete(oldestKey);
  }
  responseCache.set(key, { result, cachedAt: Date.now() });
}

const PAGE_SIZE = 30;
// A sane upper bound on how deep pagination can go. Each shard query
// below runs `ORDER BY name LIMIT (offset + PAGE_SIZE)` so the merge-sort
// across shards is provably correct (see the comment on the LIMIT below),
// but that means the per-shard LIMIT grows with the page number — capped
// here so a malicious/accidental page=999999999 can't force an
// unbounded per-shard query. 5,000 pages (150,000 rows deep) is far
// beyond what anyone browsing a filtered list by hand would ever reach.
const MAX_PAGE = 5000;

function buildWhere({ categories, countries, regions, city, noWebsiteOnly, statuses }) {
  const clauses = [];
  const params = [];

  // mysql2 expands a single `?` bound to an array into a parenthesized,
  // comma-separated, correctly-escaped list — `IN (?)` with an array param
  // is the direct MySQL/TiDB equivalent of Postgres's `= ANY($1::text[])`.
  if (categories && categories.length) {
    params.push(categories);
    clauses.push(`category IN (?)`);
  }
  if (countries && countries.length) {
    params.push(countries);
    clauses.push(`country IN (?)`);
  }
  if (regions && regions.length) {
    params.push(regions);
    clauses.push(`region IN (?)`);
  }
  if (city && city.trim()) {
    // MySQL/TiDB's default collation (utf8mb4_general_ci / _0900_ai_ci) is
    // already case-insensitive, so plain LIKE is ILIKE's direct equivalent
    // here — no separate case-insensitive operator needed.
    params.push(`%${city.trim()}%`);
    clauses.push(`city LIKE ?`);
  }
  if (noWebsiteOnly) {
    // has_website is the generated+indexed column added in
    // database/schema.mysql.sql specifically so this filter (previously an
    // un-indexable `WHERE website IS NULL OR trim(website) = ''`) can use
    // a real index instead of a full scan.
    clauses.push(`has_website = 0`);
  }
  if (statuses && statuses.length) {
    params.push(statuses);
    clauses.push(`operating_status IN (?)`);
  }

  const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
  return { where, params };
}

/** Real, paginated, filtered company search. Returns { total, page,
 * pageSize, companies } where `companies` is EXACTLY the requested page
 * (never padded/estimated) and `total` is a real count(*) across every
 * shard for the same filter set.
 *
 * Cross-shard pagination correctness: each shard independently returns
 * its own top (offset + pageSize) rows sorted by name. The GLOBAL top
 * (offset + pageSize) rows by name must be a subset of the union of
 * every shard's own top (offset + pageSize) rows — standard top-k-merge
 * reasoning, the same technique tradeAreaFanOut already uses for
 * distance-sorted results. Concatenating and re-sorting that bounded
 * union, then slicing the requested page out of it, is therefore exact,
 * not an approximation — it just does more work per shard as the page
 * number grows (bounded by MAX_PAGE above). */
async function searchCompanies(filters, page) {
  const pageNum = Math.max(1, Math.min(MAX_PAGE, parseInt(page, 10) || 1));
  const key = cacheKey(filters, pageNum);
  const cached = getCached(key);
  if (cached) return cached;

  const { where, params } = buildWhere(filters);
  const offset = (pageNum - 1) * PAGE_SIZE;
  const limitPerShard = offset + PAGE_SIZE;

  const [countRows, dataRows] = await Promise.all([
    queryEachShard(`SELECT count(*) AS cnt FROM businesses_readable ${where}`, params),
    queryEachShard(
      // MySQL/TiDB has no NULLS LAST modifier; `(name IS NULL)` evaluates to
      // 0 for a real name and 1 for a null one, so ordering by that first
      // puts every null-name row after every real one — the same effect
      // Postgres's `ORDER BY name ASC NULLS LAST` had.
      `SELECT id, name, category, address, city, region, country, phone, website, operating_status
       FROM businesses_readable ${where}
       ORDER BY (name IS NULL), name ASC
       LIMIT ${limitPerShard}`,
      params
    ),
  ]);

  let total = 0;
  // mysql2 can return COUNT(*) as a JS BigInt/string depending on the
  // column's exact type — Number(...) normalizes either back to a plain
  // number before summing across shards.
  for (const { rows: [row] } of countRows) total += (row && Number(row.cnt)) || 0;

  const merged = dataRows
    .flatMap((s) => s.rows)
    .sort((a, b) => (a.name || "").localeCompare(b.name || ""));
  const companies = merged.slice(offset, offset + PAGE_SIZE);

  const result = { total, page: pageNum, pageSize: PAGE_SIZE, companies };
  setCached(key, result);
  return result;
}

module.exports = { searchCompanies, PAGE_SIZE, MAX_PAGE };
