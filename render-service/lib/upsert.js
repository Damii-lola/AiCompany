"use strict";
// Shared normalized-schema upsert logic for the Node side (only used by
// the rarely-hit /api/ingest/businesses fallback path for the Kaggle
// notebook scanner — the primary path, scan_worker.py, talks to Postgres
// directly and has its own equivalent in Python). Same approach as the
// Python scanner: resolve category/location/brand/status text to the
// small lookup tables' integer ids with bulk, batched statements instead
// of one query per value, then upsert businesses with resolved ids.

const { pool } = require("./db");

async function ensureLookupIds(client, table, col, values) {
  const unique = [...new Set(values.filter((v) => v))];
  const ids = {};
  if (unique.length === 0) return ids;
  await client.query(
    `INSERT INTO ${table} (${col}) SELECT * FROM unnest($1::text[]) ON CONFLICT (${col}) DO NOTHING`,
    [unique]
  );
  const { rows } = await client.query(`SELECT ${col} AS v, id FROM ${table} WHERE ${col} = ANY($1)`, [unique]);
  for (const r of rows) ids[r.v] = r.id;
  return ids;
}

async function ensureLocationIds(client, triples) {
  const key = (t) => t.join("\u0001");
  const uniqueMap = new Map(triples.map((t) => [key(t), t]));
  const unique = [...uniqueMap.values()];
  const ids = {};
  if (unique.length === 0) return ids;
  await client.query(
    `INSERT INTO locations (city, region, country)
     SELECT * FROM unnest($1::text[], $2::text[], $3::text[])
     ON CONFLICT (city, region, country) DO NOTHING`,
    [unique.map((t) => t[0]), unique.map((t) => t[1]), unique.map((t) => t[2])]
  );
  const { rows } = await client.query(
    `SELECT l.city, l.region, l.country, l.id
     FROM locations l
     JOIN (SELECT unnest($1::text[]) AS city, unnest($2::text[]) AS region, unnest($3::text[]) AS country) t
       ON l.city = t.city AND l.region = t.region AND l.country = t.country`,
    [unique.map((t) => t[0]), unique.map((t) => t[1]), unique.map((t) => t[2])]
  );
  for (const r of rows) ids[key([r.city, r.region, r.country])] = r.id;
  return ids;
}

const norm = (v) => (v === null || v === undefined ? "" : String(v));

async function upsertBusinesses(rows) {
  if (rows.length === 0) return 0;
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const categoryIds = await ensureLookupIds(
      client, "categories", "name",
      rows.flatMap((r) => [norm(r.category), norm(r.category_extra)])
    );
    const brandIds = await ensureLookupIds(client, "brands", "name", rows.map((r) => norm(r.brand)));
    const statusIds = await ensureLookupIds(
      client, "statuses", "name", rows.map((r) => norm(r.operating_status))
    );
    const locationIds = await ensureLocationIds(
      client, rows.map((r) => [norm(r.city), norm(r.region), norm(r.country) || "US"])
    );
    const locKey = (r) => [norm(r.city), norm(r.region), norm(r.country) || "US"].join("\u0001");

    const now = new Date().toISOString();
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

    const placeholders = values
      .map((_, i) => `(${cols.map((_c, j) => `$${i * cols.length + j + 1}`).join(",")})`)
      .join(",");
    const updateSet = cols.filter((c) => c !== "id").map((c) => `${c} = excluded.${c}`).join(", ");
    await client.query(
      `INSERT INTO businesses (${cols.join(",")}) VALUES ${placeholders}
       ON CONFLICT (id) DO UPDATE SET ${updateSet}`,
      values.flat()
    );

    await client.query("COMMIT");
    return rows.length;
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

module.exports = { upsertBusinesses };
