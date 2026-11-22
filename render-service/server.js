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
const {
  parseRequest, resolveWithContext, resolveModelIntent, decideDeliveryMode, buildChatReply, buildCsv, streamPdf,
  isCountQuery, isNearMeQuery, extractCountSubject, buildCountReply,
} = require("./lib/chat");
const { understandWithModel } = require("./lib/model_client");

// Shared by both the model-driven and deterministic-fallback SEARCH paths
// in /api/chat — once a request has been resolved to a real address (or
// carried-over lat/lon) + real matched categories + radius, geocoding,
// the actual trade-area query, and delivery-format branching are
// identical regardless of which understanding layer produced `resolved`.
async function handleResolvedSearch({ res, resolved, country, mode, unsupportedNote }) {
  let { lat, lon } = resolved;
  if (resolved.needsGeocode) {
    ({ lat, lon } = await geocodeAddress(resolved.address));
  }

  const { businesses, summary } = await tradeAreaFanOut(
    lat, lon, resolved.radiusMiles, resolved.matchedCategories, resolved.noWebsiteOnly, country
  );

  const payload = {
    address: resolved.address, matchedCategories: resolved.matchedCategories,
    radiusMiles: resolved.radiusMiles, summary, businesses, noWebsiteOnly: resolved.noWebsiteOnly,
  };
  // Carried back to the client on EVERY response (JSON or file) so the
  // next turn in this chat — even right after a CSV/PDF download — can
  // resolve a follow-up against it. This is the chat's only "memory":
  // render-service itself holds no server-side session state, the
  // client's own conversation history is the source of truth.
  const nextContext = {
    address: resolved.address, lat, lon, matchedCategories: resolved.matchedCategories,
    radiusMiles: resolved.radiusMiles, noWebsiteOnly: resolved.noWebsiteOnly,
  };

  if (mode === "csv" || mode === "xlsx") {
    // xlsx would need a real spreadsheet-writing dependency (exceljs) —
    // not added yet, so fall back to CSV (still a real, correct file,
    // just not the exact extension asked for) rather than silently
    // returning JSON when a file was explicitly requested.
    res.setHeader("Content-Type", "text/csv");
    res.setHeader("Content-Disposition", 'attachment; filename="trade_area_snapshot.csv"');
    res.setHeader("X-Chat-Context", Buffer.from(JSON.stringify(nextContext)).toString("base64"));
    res.setHeader("Access-Control-Expose-Headers", "X-Chat-Context, Content-Disposition");
    return res.send(buildCsv(businesses));
  }
  if (mode === "pdf") {
    res.setHeader("X-Chat-Context", Buffer.from(JSON.stringify(nextContext)).toString("base64"));
    res.setHeader("Access-Control-Expose-Headers", "X-Chat-Context, Content-Disposition");
    return streamPdf(res, payload);
  }

  let reply = buildChatReply(payload);
  if (unsupportedNote) reply += `\n\n(${unsupportedNote})`;
  return res.json({
    mode: "chat",
    reply,
    matched_categories: resolved.matchedCategories,
    center: { lat, lon },
    radius_miles: resolved.radiusMiles,
    summary,
    businesses,
    context: nextContext,
  });
}

// Shared by both the model-driven and deterministic-fallback count paths
// in /api/chat — a real nationwide (no radius) count query against
// whichever real categories were matched (or every business, if none).
async function runCountQuery(matchedCategories) {
  const where = matchedCategories.length ? "WHERE category = ANY($1)" : "";
  const params = matchedCategories.length ? [matchedCategories] : [];
  const perShard = await queryEachShard(
    `SELECT count(*)::int AS total,
            count(*) FILTER (WHERE website IS NULL OR website = '')::int AS no_website
     FROM businesses_readable ${where}`,
    params
  );
  let total = 0, noWebsite = 0;
  for (const { rows: [row] } of perShard) {
    total += row.total || 0;
    noWebsite += row.no_website || 0;
  }
  return { total, noWebsite };
}

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
    const { message, context } = req.body || {};
    if (!message || typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ error: "Body must include a non-empty 'message'." });
    }

    const realCategories = await getRealCategories();

    // --- Primary understanding layer: OUR OWN fine-tuned worker model ---
    // (base Qwen2.5-7B-Instruct + template adapter + this worker's own
    // trained adapter — see workers/researchers-company-scrapper-chat/).
    // Returns null (not an error) if MODEL_SERVER_URL isn't set yet, the
    // model is cold-starting/unreachable, or its output didn't parse —
    // in every one of those cases execution falls straight through to
    // the deterministic regex pipeline below, unchanged, so a user never
    // sees a gap in service while the model is being trained/deployed.
    const modelIntent = await understandWithModel(message, context);
    if (modelIntent) {
      if (modelIntent.intent === "clarify" || modelIntent.intent === "unsupported") {
        return res.json({
          mode: "chat",
          reply: modelIntent.clarifying_reply || modelIntent.unsupported_note
            || "Could you say a bit more about what you're looking for?",
          context: context || null,
        });
      }

      const freshCategories = modelIntent.category_phrase
        ? closeMatches(modelIntent.category_phrase, realCategories) : [];

      if (modelIntent.intent === "count") {
        const matchedCategories = freshCategories.length ? freshCategories
          : (context && context.matchedCategories) || [];
        const { total, noWebsite } = await runCountQuery(matchedCategories);
        let reply = buildCountReply({ matchedCategories, total, noWebsite, ignoredWords: [] });
        if (modelIntent.unsupported_note) reply += `\n\n(${modelIntent.unsupported_note})`;
        return res.json({
          mode: "chat", reply, matched_categories: matchedCategories, total, no_website_count: noWebsite,
        });
      }

      // intent === "search"
      const resolved = resolveModelIntent(modelIntent, freshCategories, context);
      if (resolved && resolved.matchedCategories.length > 0) {
        return await handleResolvedSearch({
          res, resolved, country: "US",
          mode: modelIntent.delivery_mode, unsupportedNote: modelIntent.unsupported_note,
        });
      }
      // Model's search couldn't actually be resolved (e.g. no address and
      // no usable context) — fall through to the deterministic pipeline
      // rather than erroring, in case it does better with this exact text.
    }

    // This platform has no device/browser location at all — "near me"
    // can never be resolved by the geocoder (it's not a real place),
    // and sending it there just produced a confusing "no match" error.
    // Caught up front, before count-detection or parsing, since a count
    // question can ask for "near me" too ("how many coffee shops near me").
    if (isNearMeQuery(message)) {
      return res.status(422).json({
        error: "I don't have access to your location — tell me a city, address, or place name instead of "
          + '"near me" (e.g. "coffee shops within 2 miles of Denver, CO").',
      });
    }

    // Count/aggregate questions ("how many coffee shops are in the
    // database") have no location at all — checked FIRST, before the
    // address-requiring trade-area path, so a locationless question
    // never gets misrouted into geocoding (which previously either
    // errored on "no address" or, worse, got handed filler text like
    // "in ur database" and produced a confusing geocoder error).
    if (isCountQuery(message)) {
      const { category: subject, ignoredWords } = extractCountSubject(message, realCategories);
      const matchedCategories = subject ? closeMatches(subject, realCategories) : [];
      const { total, noWebsite } = await runCountQuery(matchedCategories);
      return res.json({
        mode: "chat",
        reply: buildCountReply({ matchedCategories, total, noWebsite, ignoredWords }),
        matched_categories: matchedCategories,
        total, no_website_count: noWebsite,
      });
    }

    const parsed = parseRequest(message, realCategories);

    // Category matching happens up front (even for a follow-up with no
    // new address) so resolveWithContext can tell "this message DID name
    // a new category" (e.g. "what about pharmacies instead") apart from
    // "this message named nothing new" (e.g. "list all of them") — the
    // latter is the exact case that used to misfire into a geocode call
    // on garbage text. closeMatches("") always returns [] cleanly, so
    // this is safe to call even when parsed.category is empty.
    const freshCategories = parsed.category ? closeMatches(parsed.category, realCategories) : [];

    const resolved = resolveWithContext(parsed, freshCategories, context);
    if (!resolved) {
      return res.status(422).json({
        error: "Couldn't find a location in that request — include an address, city, or place name "
          + "(e.g. \"coffee shops with no website within 2 miles of 123 Main St, Columbus, OH\"). "
          + "If you meant to follow up on an earlier search, that context wasn't included with this request.",
      });
    }
    if (resolved.matchedCategories.length === 0) {
      return res.status(422).json({
        error: `Could not match "${parsed.category}" to any real category in the live database — `
          + `try a different or more specific term rather than searching everything.`,
        real_category_sample: realCategories.slice(0, 20),
      });
    }

    const mode = decideDeliveryMode(message);
    return await handleResolvedSearch({ res, resolved, country: parsed.country, mode, unsupportedNote: null });
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
