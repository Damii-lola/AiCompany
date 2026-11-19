"use strict";
require("dotenv").config();

const express = require("express");
const cors = require("cors");

const { query } = require("./lib/db");
const { getRealCategories, closeMatches } = require("./lib/categories");
const { geocodeAddress } = require("./lib/geocode");
const { upsertBusinesses } = require("./lib/upsert");

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
app.get("/api/stats", async (_req, res) => {
  try {
    const [{ total }] = await query("SELECT count(*)::int AS total FROM businesses");
    const [{ no_website }] = await query(
      "SELECT count(*)::int AS no_website FROM businesses WHERE website IS NULL OR website = ''"
    );
    const [{ last_updated }] = await query(
      "SELECT max(last_seen_at) AS last_updated FROM businesses"
    );
    const categories = await getRealCategories();

    res.json({
      total_businesses: total || 0,
      no_website_count: no_website || 0,
      distinct_categories: categories.length,
      last_updated: last_updated || null,
    });
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
      where.push(`category = ANY($${params.length})`);
    }
    if (country) {
      params.push(country);
      where.push(`country = $${params.length}`);
    }
    if (region) {
      params.push(region);
      where.push(`region = $${params.length}`);
    }
    if (city) {
      params.push(city);
      where.push(`city = $${params.length}`);
    }
    if (no_website === "true") {
      where.push(`(website IS NULL OR website = '')`);
    }
    params.push(Math.min(parseInt(limit, 10) || 100, 1000));
    const sql = `SELECT * FROM businesses_readable
                 ${where.length ? "WHERE " + where.join(" AND ") : ""}
                 LIMIT $${params.length}`;
    const rows = await query(sql, params);
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
    const [rows, summary] = await Promise.all([
      query(
        "SELECT * FROM businesses_within_radius($1, $2, $3, $4, $5, $6)",
        [parseFloat(lat), parseFloat(lon), parseFloat(radius_miles), categories,
          no_website_only === "true", country || null]
      ),
      query(
        "SELECT * FROM businesses_trade_area_summary($1, $2, $3, $4, $5)",
        [parseFloat(lat), parseFloat(lon), parseFloat(radius_miles), categories, country || null]
      ),
    ]);
    res.json({ summary: summary?.[0] || null, businesses: rows });
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
    const [summaryRows, businesses] = await Promise.all([
      query(
        "SELECT * FROM businesses_trade_area_summary($1, $2, $3, $4, $5)",
        [lat, lon, radiusMiles, categories, country || null]
      ),
      query(
        "SELECT * FROM businesses_within_radius($1, $2, $3, $4, $5, $6)",
        [lat, lon, radiusMiles, categories, !!no_website_only, country || null]
      ),
    ]);

    res.json({
      address, category,
      matched_categories: categories,
      center: { lat, lon },
      radius_miles: radiusMiles,
      summary: summaryRows?.[0] || null,
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
