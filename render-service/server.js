"use strict";
require("dotenv").config();

const express = require("express");
const cors = require("cors");

const { shards, queryEachShard, queryAll } = require("./lib/db");
const { getRealCategories, closeMatches } = require("./lib/categories");
const { geocodeAddress } = require("./lib/geocode");
const { getFilterOptions } = require("./lib/filters");
const { searchCompanies } = require("./lib/companies");
const { upsertBusinesses } = require("./lib/upsert");
// TiDB has no stored functions/procedures and no spatial types, so the
// radius/"trade area" fan-out-across-shards-then-merge logic that used to
// be two Postgres SQL functions (businesses_within_radius/
// businesses_trade_area_summary) is now real application code — see
// lib/trade-area.js for the bounding-box-prefilter + real-haversine-
// distance implementation, shared by /api/trade-area and /api/research.
const { tradeAreaFanOut } = require("./lib/trade-area");

const app = express();
app.use(express.json({ limit: "10mb" })); // ingest batches can be sizeable
app.use(cors()); // every route here serves/accepts public, non-personal business data; the one
                  // sensitive route (/api/ingest) is locked by a shared secret below, not by CORS

const PORT = process.env.PORT || 3000;
const INGEST_SECRET = process.env.INGEST_SECRET;

// ---------------------------------------------------------------------
// Health check — what Render itself polls to know the service is alive
// ---------------------------------------------------------------------
app.get("/health", (_req, res) => res.json({ ok: true, time: new Date().toISOString() }));

// ---------------------------------------------------------------------
// Live stats — what the GitHub Pages dashboard shows
// ---------------------------------------------------------------------
// Cached briefly: this is an unfiltered count(*) across every shard's
// entire businesses table (millions of rows each) and was previously
// re-run from scratch on EVERY page load, by EVERY visitor, with zero
// caching — a direct contributor to the page feeling slow. The real
// numbers only move when scan_worker.py's 6-hour cron writes new rows,
// so a 60-second-old real count is still a true, current answer.
let statsCache = { data: null, fetchedAt: 0 };
const STATS_CACHE_TTL_MS = 60 * 1000;

app.get("/api/stats", async (_req, res) => {
  try {
    if (statsCache.data && Date.now() - statsCache.fetchedAt < STATS_CACHE_TTL_MS) {
      return res.json(statsCache.data);
    }

    // One shard's count is meaningless on its own — sum every shard's
    // count for a real total, and take the latest last_seen_at across
    // all of them (a business synced into shard 2 ten minutes ago still
    // counts as "last updated" even if shard 1 hasn't been touched today).
    const [perShard, categories] = await Promise.all([
      queryEachShard(`
        SELECT count(*) AS total,
               SUM(CASE WHEN has_website = 0 THEN 1 ELSE 0 END) AS no_website,
               max(last_seen_at) AS last_updated
        FROM businesses
      `),
      // MySQL/TiDB has no FILTER clause (Postgres-only) — the has_website
      // generated+indexed column (database/schema.mysql.sql) plus a plain
      // SUM(CASE WHEN ...) is the direct equivalent, and it's indexed too.
      getRealCategories(),
    ]);
    let total = 0, no_website = 0, last_updated = null;
    for (const { rows: [row] } of perShard) {
      total += Number(row.total) || 0;
      no_website += Number(row.no_website) || 0;
      if (row.last_updated && (!last_updated || row.last_updated > last_updated)) {
        last_updated = row.last_updated;
      }
    }

    const data = {
      total_businesses: total,
      no_website_count: no_website,
      distinct_categories: categories.length,
      last_updated,
      shard_count: shards.length,
    };
    statsCache = { data, fetchedAt: Date.now() };
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/categories", async (_req, res) => {
  try {
    res.json({ categories: await getRealCategories() });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------------------------------------------------------------------
// "Find companies" browse UI (Apollo-style: pick real filters from
// checkboxes, see a paginated real result list) — these two routes back
// it. /api/filters gives the UI every real, pickable filter VALUE (live,
// never hardcoded); /api/companies runs the actual filtered+paginated
// query. Both are plain GETs (filters are small enough for a query
// string) so a filtered page is a shareable/bookmarkable/back-button-able
// URL, same as Apollo's own "Find companies" page.
// ---------------------------------------------------------------------
app.get("/api/filters", async (_req, res) => {
  try {
    const [categories, { countries, regions, statuses }] = await Promise.all([
      getRealCategories(), getFilterOptions(),
    ]);
    res.json({ categories, countries, regions, statuses });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/companies", async (req, res) => {
  try {
    const splitParam = (v) => (v ? String(v).split(",").map((s) => s.trim()).filter(Boolean) : []);
    const filters = {
      categories: splitParam(req.query.categories),
      countries: splitParam(req.query.countries),
      regions: splitParam(req.query.regions),
      statuses: splitParam(req.query.statuses),
      city: req.query.city ? String(req.query.city) : "",
      noWebsiteOnly: req.query.no_website_only === "true" || req.query.no_website_only === "1",
    };
    const result = await searchCompanies(filters, req.query.page);
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------------------------------------------------------------------
// Plain exact-match search (category/country/region/city/no_website) —
// the non-radius path, mirroring service_lead_research_report.py.
// Reads through businesses_readable (database/schema.sql) so results
// come back as real text (category/city/region/country names), not the
// integer ids the underlying businesses table actually stores.
// ---------------------------------------------------------------------
app.get("/api/businesses", async (req, res) => {
  try {
    const { category, country, region, city, no_website, limit } = req.query;
    const where = [];
    const params = [];
    if (category) {
      params.push(String(category).split(","));
      where.push(`category IN (?)`); // mysql2 expands an array param into a real IN-list
    }
    if (country) {
      params.push(country);
      where.push(`country = ?`);
    }
    if (region) {
      params.push(region);
      where.push(`region = ?`);
    }
    if (city) {
      params.push(city);
      where.push(`city = ?`);
    }
    if (no_website === "true") {
      where.push(`has_website = 0`);
    }
    // Ask each shard for up to the full limit (not limit/N) — a business
    // list has no meaningful cross-shard order to begin with, and asking
    // each shard for fewer than the full limit risks returning less than
    // `limit` total even when enough real rows exist, just unevenly
    // spread across shards. Trim to the real limit after merging instead.
    const capped = Math.min(parseInt(limit, 10) || 100, 1000);
    const sql = `SELECT * FROM businesses_readable
                 ${where.length ? "WHERE " + where.join(" AND ") : ""}
                 LIMIT ${capped}`;
    const rows = (await queryAll(sql, params)).slice(0, capped);
    res.json({ count: rows.length, businesses: rows });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------------------------------------------------------------------
// Radius ("trade area") search — backed by the real Postgres functions in
// database/schema.sql (unchanged signatures from the Supabase version).
// ---------------------------------------------------------------------
app.get("/api/trade-area", async (req, res) => {
  try {
    const { lat, lon, radius_miles, category, no_website_only, country } = req.query;
    if (!lat || !lon || !radius_miles || !category) {
      return res.status(400).json({
        error: "lat, lon, radius_miles, and category are all required query params.",
      });
    }
    const categories = String(category).split(",");
    const { businesses, summary } = await tradeAreaFanOut(
      parseFloat(lat), parseFloat(lon), parseFloat(radius_miles),
      categories, no_website_only === "true", country
    );
    res.json({ summary, businesses });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------------------------------------------------------------------
// Research: structured fields in (address, category, radius, no-website
// toggle) -> real query out. Deliberately NO LLM call anywhere in this
// path. Geocoding is real (Nominatim), category matching is real (fuzzy
// string match against the LIVE distinct category list, lib/categories.js)
// — both deterministic, both verifiable, neither one guessing.
// ---------------------------------------------------------------------
app.post("/api/research", async (req, res) => {
  try {
    const { address, category, radius_miles, no_website_only, country } = req.body || {};
    if (!address || typeof address !== "string") {
      return res.status(400).json({ error: "Body must include 'address' (a real street address or place name)." });
    }
    if (!category || typeof category !== "string") {
      return res.status(400).json({ error: "Body must include 'category' (e.g. 'dental clinic', 'coffee shop')." });
    }

    const { lat, lon } = await geocodeAddress(address);

    const realCategories = await getRealCategories();
    const categories = closeMatches(category, realCategories);
    if (categories.length === 0) {
      return res.status(422).json({
        error: `Could not match "${category}" to any real category in the live database — `
          + `try a different or more specific term rather than searching everything.`,
        real_category_sample: realCategories.slice(0, 20),
      });
    }

    const radiusMiles = Number(radius_miles) > 0 ? Number(radius_miles) : 3;
    const { businesses, summary } = await tradeAreaFanOut(
      lat, lon, radiusMiles, categories, !!no_website_only, country
    );

    res.json({
      address, category,
      matched_categories: categories,
      center: { lat, lon },
      radius_miles: radiusMiles,
      summary,
      businesses,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------------------------------------------------------------------
// Ingest — the fallback write path for the optional Kaggle manual
// scanner (the primary path, scan_worker.py, writes to Postgres
// directly). Guarded by a shared secret (never a database credential —
// Kaggle never gets one) so this endpoint can be public without being an
// open write door. Resolves category/location/brand/status text to the
// normalized schema's integer ids via lib/upsert.js.
// ---------------------------------------------------------------------
app.post("/api/ingest/businesses", async (req, res) => {
  try {
    const provided = req.get("x-ingest-secret");
    if (!INGEST_SECRET || provided !== INGEST_SECRET) {
      return res.status(401).json({ error: "Missing or incorrect x-ingest-secret header." });
    }
    const { businesses } = req.body || {};
    if (!Array.isArray(businesses) || businesses.length === 0) {
      return res.status(400).json({ error: "Body must include a non-empty 'businesses' array." });
    }
    const rows = businesses.filter((b) => b && b.id); // id is the real Overture dedupe key
    if (rows.length === 0) {
      return res.status(400).json({ error: "No rows had a real 'id' field — nothing to ingest." });
    }
    const upserted = await upsertBusinesses(rows);
    res.json({ upserted });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`Company Scrapper service listening on port ${PORT}`);
});
