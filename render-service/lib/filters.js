"use strict";
// Real, live filter OPTIONS for the "Find companies" browse UI (Apollo-
// style checkbox sidebar) — country/region/operating-status values
// actually present in the database, never a hand-maintained guess list.
// Categories are already covered by lib/categories.js's getRealCategories()
// and reused as-is here; this file only adds the location/status side.
//
// Cached the same way as categories.js (5 min TTL) for the same reason:
// these are small, slow-changing lookup tables, not worth a fresh
// cross-shard scan on every page load of the filter sidebar.

const { queryEachShard } = require("./db");

let cache = { options: null, fetchedAt: 0 };
const CACHE_TTL_MS = 5 * 60 * 1000;

async function getFilterOptions() {
  const fresh = Date.now() - cache.fetchedAt < CACHE_TTL_MS;
  if (fresh && cache.options) return cache.options;

  const [countryRows, regionRows, statusRows] = await Promise.all([
    queryEachShard("SELECT DISTINCT country FROM locations WHERE country IS NOT NULL AND country <> ''"),
    queryEachShard("SELECT DISTINCT region, country FROM locations WHERE region IS NOT NULL AND region <> ''"),
    queryEachShard("SELECT name FROM statuses WHERE name IS NOT NULL AND name <> ''"),
  ]);

  const countries = new Set();
  for (const { rows } of countryRows) for (const r of rows) countries.add(r.country);

  // Regions are grouped by country (a region name like "OH" only makes
  // sense in context — nothing stops two countries from reusing a code),
  // even though today every real row is US/SBA-territory scoped.
  const regionsByCountry = {};
  for (const { rows } of regionRows) {
    for (const r of rows) {
      if (!regionsByCountry[r.country]) regionsByCountry[r.country] = new Set();
      regionsByCountry[r.country].add(r.region);
    }
  }
  const regions = {};
  for (const [country, set] of Object.entries(regionsByCountry)) regions[country] = [...set].sort();

  const statuses = new Set();
  for (const { rows } of statusRows) for (const r of rows) statuses.add(r.name);

  cache = {
    options: { countries: [...countries].sort(), regions, statuses: [...statuses].sort() },
    fetchedAt: Date.now(),
  };
  return cache.options;
}

module.exports = { getFilterOptions };
