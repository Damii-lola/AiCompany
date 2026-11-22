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

  if (categories && categories.length) {
    params.push(categories);
    clauses.push(`category = ANY($${params.length}::text[])`);
  }
  if (countries && countries.length) {
    params.push(countries);
    clauses.push(`country = ANY($${params.length}::text[])`);
  }
  if (regions && regions.length) {
    params.push(regions);
    clauses.push(`region = ANY($${params.length}::text[])`);
  }
  if (city && city.trim()) {
    params.push(`%${city.trim()}%`);
    clauses.push(`city ILIKE $${params.length}`);
  }
  if (noWebsiteOnly) {
    clauses.push(`(website IS NULL OR trim(website) = '')`);
  }
  if (statuses && statuses.length) {
    params.push(statuses);
    clauses.push(`operating_status = ANY($${params.length}::text[])`);
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
  const { where, params } = buildWhere(filters);
  const pageNum = Math.max(1, Math.min(MAX_PAGE, parseInt(page, 10) || 1));
  const offset = (pageNum - 1) * PAGE_SIZE;
  const limitPerShard = offset + PAGE_SIZE;

  const [countRows, dataRows] = await Promise.all([
    queryEachShard(`SELECT count(*)::int AS cnt FROM businesses_readable ${where}`, params),
    queryEachShard(
      `SELECT id, name, category, address, city, region, country, phone, website, operating_status
       FROM businesses_readable ${where}
       ORDER BY name ASC NULLS LAST
       LIMIT ${limitPerShard}`,
      params
    ),
  ]);

  let total = 0;
  for (const { rows: [row] } of countRows) total += (row && row.cnt) || 0;

  const merged = dataRows
    .flatMap((s) => s.rows)
    .sort((a, b) => (a.name || "").localeCompare(b.name || ""));
  const companies = merged.slice(offset, offset + PAGE_SIZE);

  return { total, page: pageNum, pageSize: PAGE_SIZE, companies };
}

module.exports = { searchCompanies, PAGE_SIZE, MAX_PAGE };
