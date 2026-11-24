"use strict";
// Shared normalized-schema upsert logic for the Node side (only used by
// the rarely-hit /api/ingest/businesses fallback path for the Kaggle
// notebook scanner — the primary path, scan_worker.py, talks to TiDB
// directly and has its own equivalent in Python). Same approach as the
// Python scanner: resolve category/location/brand/status text to the
// small lookup tables' integer ids with bulk, batched statements instead
// of one query per value, then upsert businesses with resolved ids.
//
// Rewritten from Postgres (`unnest()`/`ON CONFLICT`) to MySQL/TiDB
// (`INSERT IGNORE` + mysql2's `VALUES ?` bulk-row expansion + `IN (?)`
// array expansion) as part of the move off Prisma Postgres to TiDB Cloud.
//
// SHARDED: `pool` from ./db is always the ACTIVE (highest-numbered
// configured DATABASE_URL_N) shard's pool — every write through this
// file lands in that one shard, never spread across shards. See lib/db.js
// for why that's the right rule for this particular write path.

const { pool } = require("./db");

async function ensureLookupIds(conn, table, col, values) {
  const unique = [...new Set(values.filter((v) => v))];
  const ids = {};
  if (unique.length === 0) return ids;
  // INSERT IGNORE skips rows that would violate the unique(name) constraint
  // instead of erroring — the direct MySQL/TiDB equivalent of Postgres's
  // `ON CONFLICT (col) DO NOTHING`. mysql2's `VALUES ?` with a nested
  // array-of-arrays expands to a real multi-row INSERT in one round trip.
  await conn.query(`INSERT IGNORE INTO ${table} (${col}) VALUES ?`, [unique.map((v) => [v])]);
  const [rows] = await conn.query(`SELECT ${col} AS v, id FROM ${table} WHERE ${col} IN (?)`, [unique]);
  for (const r of rows) ids[r.v] = r.id;
  return ids;
}

async function ensureLocationIds(conn, triples) {
  const key = (t) => t.join("\u0001");
  const uniqueMap = new Map(triples.map((t) => [key(t), t]));
  const unique = [...uniqueMap.values()];
  const ids = {};
  if (unique.length === 0) return ids;
  await conn.query(
    `INSERT IGNORE INTO locations (city, region, country) VALUES ?`,
    [unique]
  );
  // MySQL/TiDB has no clean row-value IN-list equivalent to Postgres's
  // unnest()-joined exact-triple match, so this fetches every row matching
  // ANY of the three columns independently (a superset — locations is a
  // small table, so over-fetching here is cheap) and filters down to exact
  // triple matches in JS, which is still correct, just done client-side
  // instead of in SQL.
  const cities = unique.map((t) => t[0]);
  const regions = unique.map((t) => t[1]);
  const countries = unique.map((t) => t[2]);
  const [rows] = await conn.query(
    `SELECT city, region, country, id FROM locations
     WHERE city IN (?) AND region IN (?) AND country IN (?)`,
    [cities, regions, countries]
  );
  const wanted = new Set(unique.map(key));
  for (const r of rows) {
    const k = key([r.city, r.region, r.country]);
    if (wanted.has(k)) ids[k] = r.id;
  }
  return ids;
}

const norm = (v) => (v === null || v === undefined ? "" : String(v));

// MySQL/TiDB, like Postgres, rejects a text value containing a NUL (0x00)
// byte — strip it rather than let one bad field abort the whole batch.
// Same fix as scan_worker.py's primary path, applied here too since this
// endpoint takes arbitrary POSTed data from the Kaggle fallback scanner.
function stripNul(v) {
  return typeof v === "string" ? v.replace(/\u0000/g, "") : v;
}

async function upsertBusinesses(rawRows) {
  if (rawRows.length === 0) return 0;
  const rows = rawRows.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, stripNul(v)])));
  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();

    const categoryIds = await ensureLookupIds(
      conn, "categories", "name",
      rows.flatMap((r) => [norm(r.category), norm(r.category_extra)])
    );
    const brandIds = await ensureLookupIds(conn, "brands", "name", rows.map((r) => norm(r.brand)));
    const statusIds = await ensureLookupIds(
      conn, "statuses", "name", rows.map((r) => norm(r.operating_status))
    );
    const locationIds = await ensureLocationIds(
      conn, rows.map((r) => [norm(r.city), norm(r.region), norm(r.country) || "US"])
    );
    const locKey = (r) => [norm(r.city), norm(r.region), norm(r.country) || "US"].join("\u0001");

    const now = new Date();
    const cols = [
      "id", "name", "category_id", "category_extra_id", "address", "location_id",
      "postcode", "phone", "website", "email", "social", "brand_id", "confidence",
      "status_id", "lon", "lat", "last_seen_at",
    ];
    const values = rows.map((r) => [
      r.id, r.name ?? null,
      categoryIds[norm(r.category)] ?? null, categoryIds[norm(r.category_extra)] ?? null,
      r.address ?? null, locationIds[locKey(r)] ?? null,
      r.postcode ?? null, r.phone ?? null, r.website ?? null, r.email ?? null, r.social ?? null,
      brandIds[norm(r.brand)] ?? null, r.confidence ?? null, statusIds[norm(r.operating_status)] ?? null,
      r.lon ?? null, r.lat ?? null, now,
    ]);

    // Same philosophy as scan_worker.py: a business already in the table is
    // left alone, not silently overwritten by a later re-sighting — only
    // last_seen_at bumps, so a future targeted updater still has a real
    // "not re-confirmed since X" signal to work from. MySQL/TiDB's
    // `ON DUPLICATE KEY UPDATE` is the direct equivalent of Postgres's
    // `ON CONFLICT (id) DO UPDATE SET ...`.
    await conn.query(
      `INSERT INTO businesses (${cols.join(",")}) VALUES ?
       ON DUPLICATE KEY UPDATE last_seen_at = VALUES(last_seen_at)`,
      [values]
    );

    await conn.commit();
    return rows.length;
  } catch (e) {
    await conn.rollback();
    throw e;
  } finally {
    conn.release();
  }
}

module.exports = { upsertBusinesses };
