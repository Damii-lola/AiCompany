"use strict";
// Real category list, pulled LIVE from Postgres (never hardcoded/invented —
// mirrors company_scrapper_ai.py's REAL_CATEGORIES, but sourced from the
// live, continuously-growing table). Cached briefly so every request to
// /api/research doesn't re-scan the whole table.
//
// Backed by a real SELECT DISTINCT against the small `categories` lookup
// table directly — the old Supabase version had to pull a 50,000-row
// sample of raw text and dedupe client-side because PostgREST has no
// native DISTINCT. A direct Postgres connection doesn't have that
// limitation, and the categories table is tiny (one row per distinct
// category, not per business), so this is both simpler and exact.
//
// SHARDED: each shard has its OWN categories table with its OWN integer
// ids — the same category name ("coffee shop") can be id 7 in shard 1
// and id 3 in shard 2. That's fine here because this only ever returns
// NAMES (never ids) to callers — fan out to every shard, union the
// distinct name strings, done.

const { queryEachShard } = require("./db");

let cache = { categories: [], fetchedAt: 0 };
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes — real data changes slowly enough that this is safe

async function getRealCategories() {
  const fresh = Date.now() - cache.fetchedAt < CACHE_TTL_MS;
  if (fresh && cache.categories.length > 0) return cache.categories;

  const perShard = await queryEachShard(
    "SELECT name FROM categories WHERE name IS NOT NULL AND name <> ''"
  );
  const names = new Set();
  for (const { rows } of perShard) {
    for (const r of rows) names.add(r.name);
  }
  cache = { categories: [...names].sort(), fetchedAt: Date.now() };
  return cache.categories;
}

/** Edit-distance ratio (Levenshtein-based) — kept only as a heavily-
 * discounted fallback inside categoryScore below, for genuine near-miss
 * spelling ("vetrinary" -> "veterinary"). NEVER used alone to decide a
 * match: two unrelated category phrases of similar length can score
 * deceptively high on character similarity alone just from sharing
 * letters/spacing (this bit a previous version of this exact function —
 * "coffee shops" scored close enough to "auto_repair_shop" to return
 * both), which silently broadens a client's trade-area results to
 * include a category they never asked about. */
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

/** Crude but effective for this domain: handles the two common English
 * plural shapes separately rather than one blind rule — a trailing "ies"
 * singularizes to "y" ("pharmacies" -> "pharmacy"), which a plain
 * trailing-"s" strip would mangle into "pharmacie" and miss entirely;
 * everything else over 3 letters just drops a trailing "s"
 * ("shops" -> "shop"). */
function singularizeWord(w) {
  if (w.endsWith("ies") && w.length > 4) return w.slice(0, -3) + "y";
  if (w.endsWith("s") && w.length > 3) return w.slice(0, -1);
  return w;
}

function singularizeWords(s) {
  return new Set(s.split(/\s+/).filter(Boolean).map(singularizeWord));
}

function sameSet(a, b) {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

/** Counts word-identity overlap between two word sets, but — unlike a
 * plain set intersection — also credits a simple one-letter-off typo
 * word ("coffe" for "coffee") as a match, via per-word character
 * similarity. Gated at >=0.8 and only for words of 4+ letters
 * specifically so this never degrades into the whole-phrase
 * cross-contamination bug this file already fixed once (a short,
 * generic word like "in"/"of" fuzzy-matching something unrelated) —
 * it only forgives a genuine near-miss spelling of a real word, never
 * a loose phrase-level resemblance. */
function fuzzyWordOverlap(termSet, catSet) {
  let matched = 0;
  const usedCat = new Set();
  for (const tw of termSet) {
    if (catSet.has(tw)) { matched++; continue; }
    if (tw.length < 4) continue;
    for (const cw of catSet) {
      if (usedCat.has(cw) || cw.length < 4) continue;
      if (similarity(tw, cw) >= 0.8) { matched++; usedCat.add(cw); break; }
    }
  }
  return matched;
}

/** Whole-WORD containment — is `small`'s word sequence a contiguous
 * subsequence of `big`'s words? Deliberately NOT a raw string .includes()
 * (see containsPhrase below for why that was a real bug). */
function containsWordSeq(bigWords, smallWords) {
  if (smallWords.length === 0 || smallWords.length > bigWords.length) return false;
  for (let i = 0; i <= bigWords.length - smallWords.length; i++) {
    let ok = true;
    for (let j = 0; j < smallWords.length; j++) {
      if (bigWords[i + j] !== smallWords[j]) { ok = false; break; }
    }
    if (ok) return true;
  }
  return false;
}

/** Whole-word phrase containment between two already-lowercased,
 * space-separated phrases. Replaces a raw String.includes() check that
 * was a real, live bug: "of".includes-tested against "coffee shop"
 * scored 0.9 because the LETTERS "o"+"f" happen to appear back-to-back
 * inside "coffee" — a pure substring coincidence with zero relation to
 * the actual word "of". That one bug alone let a two-letter stopword in
 * a contentless follow-up message ("list all OF them") get silently
 * treated as a confident category match. Word-boundary containment
 * ("coffee" is contained in "coffee shop", "of" is not) is the correct
 * semantics this check was always meant to have. */
function containsPhrase(bigPhrase, smallPhrase) {
  return containsWordSeq(bigPhrase.split(/\s+/), smallPhrase.split(/\s+/));
}

/** Word-identity-first scoring, NOT raw character similarity. Exact or
 * singularized-word-set matches score highest; character ratio
 * (similarity above) is kept only as a heavily-discounted fallback for
 * genuine near-miss spelling, never enough on its own to beat a real
 * word-level mismatch. Same approach as (and must stay in sync with)
 * company_scrapper_ai.py's _category_score. */
function categoryScore(term, normCat) {
  if (term === normCat) return 1;
  const termSet = singularizeWords(term);
  const catSet = singularizeWords(normCat);
  if (sameSet(termSet, catSet)) return 1;
  if (containsPhrase(normCat, term) || containsPhrase(term, normCat)) return 0.9;
  const union = new Set([...termSet, ...catSet]);
  const overlap = fuzzyWordOverlap(termSet, catSet);
  const jaccard = union.size ? overlap / union.size : 0;
  const charRatio = similarity(term, normCat);
  return Math.max(jaccard * 0.85, charRatio * 0.5);
}

/** Maps what a user actually says ("dentist", "coffee shop") to the REAL
 * category strings in the database. Scores every real category against
 * the space-normalized form (Overture's real taxonomy values are
 * typically snake_case, "dental_clinic", "coffee_shop", while a user
 * types them with spaces) via categoryScore, then — critically — only
 * returns matches within a real margin of the single best score, so a
 * weak secondary match never gets smuggled into a list that's OR'd
 * together in the live query (which previously silently broadened
 * results to include a category nobody asked for). */
function closeMatches(term, categories, topK = 3, minScore = 0.45) {
  const t = term.trim().toLowerCase();
  if (!t) return [];
  const scored = categories
    .map((c) => ({ c, score: categoryScore(t, c.replace(/_/g, " ").toLowerCase()) }))
    .filter((x) => x.score >= minScore)
    .sort((a, b) => b.score - a.score);
  if (scored.length === 0) return [];
  const best = scored[0].score;
  return scored.filter((x) => x.score >= best - 0.2).slice(0, topK).map((x) => x.c);
}

module.exports = { getRealCategories, closeMatches, categoryScore, singularizeWords };
