"use strict";
// Real category list, pulled LIVE from Postgres (never hardcoded/invented —
// mirrors company_scrapper_ai.py's REAL_CATEGORIES, but sourced from the
// live, continuously-growing table). Cached briefly so every request to
// /api/research doesn't re-scan the whole table.
//
// Now backed by a real SELECT DISTINCT against the small `categories`
// lookup table directly — the old Supabase version had to pull a 50,000-
// row sample of raw text and dedupe client-side because PostgREST has no
// native DISTINCT. A direct Postgres connection doesn't have that
// limitation, and the categories table is tiny (one row per distinct
// category, not per business), so this is now both simpler and exact.

const { query } = require("./db");

let cache = { categories: [], fetchedAt: 0 };
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes — real data changes slowly enough that this is safe

async function getRealCategories() {
  const fresh = Date.now() - cache.fetchedAt < CACHE_TTL_MS;
  if (fresh && cache.categories.length > 0) return cache.categories;

  const rows = await query(
    "SELECT name FROM categories WHERE name IS NOT NULL AND name <> '' ORDER BY name"
  );
  cache = { categories: rows.map((r) => r.name), fetchedAt: Date.now() };
  return cache.categories;
}

/** Lightweight close-match (token overlap + simple edit-distance ratio) —
 * no extra dependency needed, same role as Python's difflib.get_close_matches
 * in company_scrapper_ai.py: a cheap first pass that narrows the real list
 * down to a shortlist before the model picks from it. */
function similarity(a, b) {
  a = a.toLowerCase();
  b = b.toLowerCase();
  if (a === b) return 1;
  const la = a.length, lb = b.length;
  if (la === 0 || lb === 0) return 0;
  const dp = Array.from({ length: la + 1 }, () => new Array(lb + 1).fill(0));
  for (let i = 0; i <= la; i++) dp[i][0] = i;
  for (let j = 0; j <= lb; j++) dp[0][j] = j;
  for (let i = 1; i <= la; i++) {
    for (let j = 1; j <= lb; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  const dist = dp[la][lb];
  return 1 - dist / Math.max(la, lb);
}

function closeMatches(term, categories, topK = 15) {
  const t = term.trim().toLowerCase();
  const tokenMatches = categories.filter(
    (c) => c.toLowerCase().includes(t.replace(/\s+/g, "_")) || t.includes(c.toLowerCase().replace(/_/g, " "))
  );
  const scored = categories
    .map((c) => ({ c, score: similarity(t, c.replace(/_/g, " ")) }))
    .filter((x) => x.score >= 0.3)
    .sort((a, b) => b.score - a.score)
    .map((x) => x.c);
  return [...new Set([...tokenMatches, ...scored])].slice(0, topK);
}

module.exports = { getRealCategories, closeMatches };
