"use strict";
require("dotenv").config();

const express = require("express");
const cors = require("cors");

const { supabase } = require("./lib/supabase");
const { parseResearchRequest, matchCategories } = require("./lib/mistral");
const { getRealCategories, closeMatches } = require("./lib/categories");
const { geocodeAddress } = require("./lib/geocode");

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
    const { count: total, error: e1 } = await supabase
      .from("businesses").select("*", { count: "exact", head: true });
    if (e1) throw e1;

    const { count: noWebsite, error: e2 } = await supabase
      .from("businesses").select("*", { count: "exact", head: true })
      .or("website.is.null,website.eq.");
    if (e2) throw e2;

    const { data: latest, error: e3 } = await supabase
      .from("businesses").select("last_seen_at").order("last_seen_at", { ascending: false }).limit(1);
    if (e3) throw e3;

    const categories = await getRealCategories();

    res.json({
      total_businesses: total || 0,
      no_website_count: noWebsite || 0,
      distinct_categories: categories.length,
      last_updated: latest?.[0]?.last_seen_at || null,
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
// the non-radius path, mirroring service_lead_research_report.py
// ---------------------------------------------------------------------
app.get("/api/businesses", async (req, res) => {
  try {
    const { category, country, region, city, no_website, limit } = req.query;
    let q = supabase.from("businesses").select("*");
    if (category) q = q.in("category", String(category).split(","));
    if (country) q = q.eq("country", country);
    if (region) q = q.eq("region", region);
    if (city) q = q.eq("city", city);
    if (no_website === "true") q = q.or("website.is.null,website.eq.");
    q = q.limit(Math.min(parseInt(limit, 10) || 100, 1000));
    const { data, error } = await q;
    if (error) throw error;
    res.json({ count: data.length, businesses: data });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------------------------------------------------------------------
// Radius ("trade area") search — the capability that was missing before
// this whole build, now backed by the real Postgres function in schema.sql
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
    const [{ data: rows, error: e1 }, { data: summary, error: e2 }] = await Promise.all([
      supabase.rpc("businesses_within_radius", {
        center_lat: parseFloat(lat), center_lon: parseFloat(lon),
        radius_miles: parseFloat(radius_miles), categories,
        only_no_website: no_website_only === "true", country_filter: country || null,
      }),
      supabase.rpc("businesses_trade_area_summary", {
        center_lat: parseFloat(lat), center_lon: parseFloat(lon),
        radius_miles: parseFloat(radius_miles), categories, country_filter: country || null,
      }),
    ]);
    if (e1) throw e1;
    if (e2) throw e2;
    res.json({ summary: summary?.[0] || null, businesses: rows });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------------------------------------------------------------------
// Natural-language research request -> real structured query -> real
// results. The on-demand equivalent of company_scrapper_ai.py's research(),
// ported to Mistral since this runs on Render (no GPU) instead of a Kaggle
// GPU notebook. Every field either came from the user's own words or was
// validated against real data (real geocode, real category match, real
// row counts from Supabase) — nothing here is invented to fill a gap.
// ---------------------------------------------------------------------
const _researchRateLimiter = new Map(); // ip -> [timestamps] — cheap guard since this calls a paid API
function rateLimited(ip, maxPerMinute = 10) {
  const now = Date.now();
  const hits = (_researchRateLimiter.get(ip) || []).filter((t) => now - t < 60_000);
  hits.push(now);
  _researchRateLimiter.set(ip, hits);
  return hits.length > maxPerMinute;
}

app.post("/api/research", async (req, res) => {
  try {
    if (rateLimited(req.ip)) {
      return res.status(429).json({ error: "Too many research requests from this IP — slow down." });
    }
    const { request: nlRequest, output_formats } = req.body || {};
    if (!nlRequest || typeof nlRequest !== "string") {
      return res.status(400).json({ error: "Body must include 'request': the plain-English research request." });
    }

    const parsed = await parseResearchRequest(nlRequest);
    if (!parsed.address) {
      return res.status(422).json({
        error: "Could not find a location to anchor the trade area in that request — "
          + "ask for a specific address rather than guessing one.",
        parsed,
      });
    }

    const { lat, lon } = await geocodeAddress(parsed.address);

    const realCategories = await getRealCategories();
    const shortlist = closeMatches(parsed.category || "", realCategories);
    const categories = await matchCategories(parsed.category || "", realCategories, shortlist);
    if (categories.length === 0) {
      return res.status(422).json({
        error: `Could not match "${parsed.category}" to any real category in the live database — `
          + `ask the user to clarify rather than searching everything.`,
        parsed,
      });
    }

    const radiusMiles = parsed.radius_miles || 3;
    const { data: summaryRows, error: e1 } = await supabase.rpc("businesses_trade_area_summary", {
      center_lat: lat, center_lon: lon, radius_miles: radiusMiles,
      categories, country_filter: parsed.country || null,
    });
    if (e1) throw e1;
    const { data: businesses, error: e2 } = await supabase.rpc("businesses_within_radius", {
      center_lat: lat, center_lon: lon, radius_miles: radiusMiles,
      categories, only_no_website: !!parsed.no_website_only, country_filter: parsed.country || null,
    });
    if (e2) throw e2;

    res.json({
      request: nlRequest,
      parsed,
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
// Ingest — the ONLY write path into Supabase. Called by the Kaggle
// scanner as it finds real businesses, batched per tile. Guarded by a
// shared secret (never a Supabase key — Kaggle never gets one) so this
// endpoint can be public (Kaggle has no stable outbound IP to allowlist)
// without being an open write door.
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
    const now = new Date().toISOString();
    const rows = businesses
      .filter((b) => b && b.id) // id is the real Overture dedupe key — never upsert a row without one
      .map((b) => ({
        id: b.id, name: b.name ?? null, category: b.category ?? null,
        category_extra: b.category_extra ?? null, address: b.address ?? null,
        city: b.city ?? null, region: b.region ?? null, postcode: b.postcode ?? null,
        country: b.country ?? null, phone: b.phone ?? null, website: b.website ?? null,
        email: b.email ?? null, social: b.social ?? null, brand: b.brand ?? null,
        confidence: b.confidence ?? null, operating_status: b.operating_status ?? null,
        lon: b.lon ?? null, lat: b.lat ?? null, last_seen_at: now,
      }));
    if (rows.length === 0) {
      return res.status(400).json({ error: "No rows had a real 'id' field — nothing to ingest." });
    }
    const { error } = await supabase.from("businesses").upsert(rows, { onConflict: "id" });
    if (error) throw error;
    res.json({ upserted: rows.length });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`Company Scrapper service listening on port ${PORT}`);
});
