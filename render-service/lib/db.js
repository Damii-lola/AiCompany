"use strict";
// Prisma Postgres — now SHARDED. One free Prisma database caps out
// (2GB storage / 200k ops a month) well before a full US+territories
// backfill finishes, so the scanner (scan_worker.py) splits its writes
// across as many free databases as you give it, via DATABASE_URL_1,
// DATABASE_URL_2, DATABASE_URL_3, ... secrets. This service is the READ
// side, so it has to fan every read out across ALL of those same shards
// and merge the results in application code — externally it still looks
// like one unified database, nothing about the API's shape changes.
//
// Discovery uses the exact same DATABASE_URL_<N> convention as
// scanner-service/scan_worker.py (keep them in sync if you ever change
// one). A bare DATABASE_URL (no numeric suffix) still works as a single
// implicit shard — nothing here requires more than one database.
const { Pool } = require("pg");

function discoverShards() {
  const shards = [];
  const pattern = /^DATABASE_URL_(\d+)$/;
  for (const [key, value] of Object.entries(process.env)) {
    const m = pattern.exec(key);
    if (m && value) shards.push({ num: parseInt(m[1], 10), url: value });
  }
  shards.sort((a, b) => a.num - b.num);
  if (shards.length === 0) {
    const bare = process.env.DATABASE_URL;
    if (!bare) {
      throw new Error(
        "No database configured — set DATABASE_URL_1 (and DATABASE_URL_2, " +
        "DATABASE_URL_3, ... for more shards) as Render environment variables, " +
        "or a bare DATABASE_URL for a single-database setup. Each one is a " +
        "Prisma Postgres connection string from your Prisma dashboard " +
        "(Connect -> the postgres://...@db.prisma.io:5432/postgres?sslmode=require " +
        "string, or the pooled.db.prisma.io variant)."
      );
    }
    shards.push({ num: 1, url: bare });
  }
  return shards;
}

const shards = discoverShards().map(({ num, url }) => ({
  num,
  pool: new Pool({ connectionString: url, ssl: { rejectUnauthorized: false } }),
}));

// The Kaggle-fallback ingest endpoint (lib/upsert.js) is the only writer
// in this service, and — same rule as the scanner — new writes always go
// to the single highest-numbered configured shard, never spread across
// shards by this service. (The scanner itself uses a different rule,
// static tile partitioning, because it's doing a one-time bulk backfill
// across a known tile list; this endpoint takes arbitrary ad-hoc rows
// with no tile to partition by, so "always the newest shard" is the
// simple, correct choice here.)
const activeShard = shards[shards.length - 1];

async function query(sql, params = []) {
  // Single-shard query against the ACTIVE (write) shard — for the one
  // write path (lib/upsert.js) and anywhere a caller deliberately only
  // wants one specific shard, never for reads that should reflect the
  // whole dataset.
  const { rows } = await activeShard.pool.query(sql, params);
  return rows;
}

async function queryEachShard(sql, params = []) {
  // Fans the same read out to every shard CONCURRENTLY and returns the
  // per-shard results separately (as [{ num, rows }, ...]) so callers
  // that need a real merge — sum counts, union distinct values, re-sort
  // by a real column like distance_miles — can do that themselves rather
  // than having rows silently flattened together.
  return Promise.all(
    shards.map(async (s) => ({ num: s.num, rows: (await s.pool.query(sql, params)).rows }))
  );
}

async function queryAll(sql, params = []) {
  // Convenience wrapper over queryEachShard for the common case: just
  // union every shard's rows into one flat array, tagging each with
  // which shard it came from (_shard) in case a caller still cares.
  const perShard = await queryEachShard(sql, params);
  return perShard.flatMap((s) => s.rows.map((r) => ({ ...r, _shard: s.num })));
}

module.exports = { shards, activeShard, pool: activeShard.pool, query, queryEachShard, queryAll };
