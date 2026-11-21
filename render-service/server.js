"use strict";
require("dotenv").config();

const express = require("express");
const cors = require("cors");

const { shards, queryEachShard, queryAll } = require("./lib/db");
const { getRealCategories, closeMatches } = require("./lib/categories");
const { geocodeAddress } = require("./lib/geocode");
const { upsertBusinesses } = require("./lib/upsert");
const { parseRequest, decideDeliveryMode, buildChatReply, buildCsv, streamPdf } = require("./lib/chat");

// Shared by /api/trade-area and /api/research — both need the exact same
// fan-out-across-shards-then-merge logic, so it lives in one place.
// businesses_within_radius() now returns a real distance_miles column
// (database/schema.sql) specifically so this can merge-sort every
// shard's already-sorted results back into one correctly-ordered list,
// instead of just concatenating shard 1's closest matches followed by
// shard 2's closest matches.
async function tradeAreaFanOut(lat, lon, radiusMiles, categories, noWebsiteOnly, country) {
  const [businessRows, summaryRows] = await Promise.all([
    queryEachShard(
      "SELECT * FROM businesses_within_radius($1, $2, $3, $4, $5, $6)",
      [lat, lon, radiusMiles, categories, !!noWebsiteOnly, country || null]
    ),
    queryEachShard(
      "SELECT * FROM businesses_trade_area_summary($1, $2, $3, $4, $5)",
      [lat, lon, radiusMiles, categories, country || null]
    ),
  ]);

  const businesses = businessRows
    .flatMap((s) => s.rows)
    .sort((a, b) => a.distance_miles - b.distance_miles);

  let competitor_count = 0, no_website_count = 0, nearest_miles = null;
  for (const { rows } of summaryRows) {
    const row = rows[0];
    if (!row) continue;
    competitor_count += Number(row.competitor_count) || 0;
    no_website_count += Number(row.no_website_count) || 0;
    if (row.nearest_miles != null && (nearest_miles === null || row.nearest_miles < nearest_miles)) {
      nearest_miles = row.nearest_miles;
    }
  }

  return { businesses, summary: { competitor_count, no_website_count, nearest_miles } };
}

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
    // One shard's count is meaningless on its own — sum every shard's
    // count for a real total, and take the latest last_seen_at across
    // all of them (a business synced into shard 2 ten minutes ago still
    // counts as "last updated" even if shard 1 hasn't been touched today).
    const perShard = await queryEachShard(`
      SELECT count(*)::int AS total,
             count(*) FILTER (WHERE website IS NULL OR website = '')::int AS no_website,
             max(last_seen_at) AS last_updated
      FROM businesses
    `);
    let total = 0, no_website = 0, last_updated = null;
    for (const { rows: [row] } of perShard) {
      total += row.total || 0;
      no_website += row.no_website || 0;
      if (row.last_updated && (!last_updated || row.last_updated > last_updated)) {
        last_updated = row.last_updated;
      }
    }
    const categories = await getRealCategories();

    res.json({
      total_businesses: total,
      no_website_count: no_website,
      distinct_categories: categories.length,
      last_updated,
      shard_count: shards.length,
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
    // Ask each shard for up to the full limit (not limit/N) — a business
    // list has no meaningful cross-shard order to begin with, and asking
    // each shard for fewer than the full limit risks returning less than
    // `limit` total even when enough real rows exist, just unevenly
    // spread across shards. Trim to the real limit after merging instead.
    const capped = Math.min(parseInt(limit, 10) || 100, 1000);
    params.push(capped);
    const sql = `SELECT * FROM businesses_readable
                 ${where.length ? "WHERE " + where.join(" AND ") : ""}
                 LIMIT $${params.length}`;
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
// Live chat — the real, always-on "addon to the template AI": a user
// types a plain-English request, this parses it deterministically
// (lib/chat.js — no model, no GPU, same reasoning as the old
// company_scrapper_ai.py, just running in this process instead of a
// GitHub Actions job), matches the category against the real live list,
// geocodes, runs the exact same tradeAreaFanOut every other route uses,
// then decides FROM THE MESSAGE ITSELF whether to answer inline or hand
// back a file — docs/index.html's chat widget calls this directly and
// gets an instant response either way, nothing to trigger manually.
// ---------------------------------------------------------------------
app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body || {};
    if (!message || typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ error: "Body must include a non-empty 'message'." });
    }

    const realCategories = await getRealCategories();
    const parsed = parseRequest(message, realCategories);

    if (!parsed.address) {
      return res.status(422).json({
        error: "Couldn't find a location in that request — include an address, city, or place name "
          + "(e.g. \"coffee shops with no website within 2 miles of 123 Main St, Columbus, OH\").",
      });
    }

    const matchedCategories = closeMatches(parsed.category, realCategories);
    if (matchedCategories.length === 0) {
      return res.status(422).json({
        error: `Could not match "${parsed.category}" to any real category in the live database — `
          + `try a different or more specific term rather than searching everything.`,
        real_category_sample: realCategories.slice(0, 20),
      });
    }

    const { lat, lon } = await geocodeAddress(parsed.address);
    const radiusMiles = parsed.radius_miles || 3;
    const { businesses, summary } = await tradeAreaFanOut(
      lat, lon, radiusMiles, matchedCategories, parsed.no_website_only, parsed.country
    );

    const mode = decideDeliveryMode(message);
    const payload = {
      address: parsed.address, matchedCategories, radiusMiles, summary, businesses,
      noWebsiteOnly: parsed.no_website_only,
    };

    if (mode === "csv" || mode === "xlsx") {
      // xlsx would need a real spreadsheet-writing dependency (exceljs) —
      // not added yet, so fall back to CSV (still a real, correct file,
      // just not the exact extension asked for) rather than silently
      // returning JSON when a file was explicitly requested.
      res.setHeader("Content-Type", "text/csv");
      res.setHeader("Content-Disposition", 'attachment; filename="trade_area_snapshot.csv"');
      return res.send(buildCsv(businesses));
    }
    if (mode === "pdf") {
      return streamPdf(res, payload);
    }

    res.json({
      mode: "chat",
      reply: buildChatReply(payload),
      matched_categories: matchedCategories,
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
