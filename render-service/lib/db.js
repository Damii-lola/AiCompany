"use strict";
// TiDB Cloud Starter — SHARDED. Moved off Prisma Postgres entirely: Prisma's
// 2GB/workspace free-tier cap was already being exceeded by the live data
// (one shard alone showed 3.4GB used), and getting more Prisma shards means
// more separate email sign-ups, which ran out. TiDB Cloud Starter gives up
// to 5 free instances PER ACCOUNT (no card needed), 5GiB storage + 50M
// request units/month EACH — 25GB total under one login, no extra emails.
//
// TiDB speaks the MySQL wire protocol, not Postgres — this file (and every
// other lib/*.js + server.js) was rewritten from `pg` to `mysql2` for that
// reason. See database/schema.mysql.sql for the schema-level changes this
// forced (no stored functions, no spatial types, no cube/earthdistance —
// radius search moved to lib/trade-area.js as real application code).
//
// Discovery uses the exact same DATABASE_URL_<N> convention as before and
// as scanner-service/scan_worker.py (keep them in sync if you ever change
// one) — just now each one is a TiDB connection string
// (mysql://user:pass@host:4000/dbname) instead of a Postgres one. A bare
// DATABASE_URL (no numeric suffix) still works as a single implicit shard.
const mysql = require("mysql2/promise");

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
        "TiDB Cloud connection string from your TiDB Cloud dashboard's " +
        "'Connect' panel (mysql://USER:PASSWORD@HOST:4000/DBNAME)."
      );
    }
    shards.push({ num: 1, url: bare });
  }
  return shards;
}

// TiDB Cloud requires TLS, and its gateway uses a publicly-trusted cert
// (unlike Prisma Postgres, which needed rejectUnauthorized:false for its
// internal routing cert) — rejectUnauthorized:true is the correct, safer
// default here. mysql2's own URI parser doesn't accept an `ssl` object via
// query string, so the connection string is parsed into a real config
// object instead of string-concatenating options onto it. A local, SSL-less
// MySQL/MariaDB (used for local testing — see database/schema.mysql.sql's
// header) has no `?ssl=` param at all, so ssl stays undefined for it.
function parseConnectionUrl(url) {
  const u = new URL(url);
  const wantsSsl = (u.searchParams.get("ssl") ?? "true") !== "false";
  return {
    host: u.hostname,
    port: u.port ? parseInt(u.port, 10) : 4000, // TiDB Cloud's default port
    user: decodeURIComponent(u.username),
    password: decodeURIComponent(u.password),
    database: u.pathname.replace(/^\//, ""),
    ssl: wantsSsl ? { rejectUnauthorized: true } : undefined,
    waitForConnections: true,
    connectionLimit: 10,
    // TiDB Cloud's own docs: connections held open longer than ~30 minutes
    // can be killed unexpectedly (server restarts/maintenance), and they
    // recommend a max connection lifetime around 5 minutes. mysql2 has no
    // direct "max lifetime" knob, so idleTimeout recycles pooled
    // connections well under that window instead of letting one sit idle
    // until it goes stale and the next query on it just fails outright.
    idleTimeout: 240000, // 4 min — under TiDB's ~5 min recommendation
    enableKeepAlive: true,
    keepAliveInitialDelay: 10000,
  };
}

// Transient connection errors (TiDB killed an idle/stale connection, a
// maintenance restart, a brief network blip) show up as a thrown error from
// pool.query() even though the pool itself is healthy — mysql2 doesn't
// retry these automatically. Retrying once with a fresh connection from the
// pool turns a real but harmless hiccup into a transparent success instead
// of a 500 on whoever's loading the Find Companies page at that moment.
const TRANSIENT_CODES = new Set([
  "PROTOCOL_CONNECTION_LOST",
  "ECONNRESET",
  "ETIMEDOUT",
  "ECONNREFUSED",
  "PROTOCOL_SEQUENCE_TIMEOUT",
]);

async function queryWithRetry(pool, sql, params) {
  try {
    const [rows] = await pool.query(sql, params);
    return rows;
  } catch (err) {
    if (!TRANSIENT_CODES.has(err.code)) throw err;
    const [rows] = await pool.query(sql, params);
    return rows;
  }
}

const shards = discoverShards().map(({ num, url }) => ({
  num,
  pool: mysql.createPool(parseConnectionUrl(url)),
}));

const activeShard = shards[shards.length - 1];

async function query(sql, params = []) {
  // Single-shard query against the ACTIVE (write) shard — for the one
  // write path (lib/upsert.js) and anywhere a caller deliberately only
  // wants one specific shard, never for reads that should reflect the
  // whole dataset.
  return queryWithRetry(activeShard.pool, sql, params);
}

async function queryEachShard(sql, params = []) {
  // Fans the same read out to every shard CONCURRENTLY and returns the
  // per-shard results separately (as [{ num, rows }, ...]) so callers
  // that need a real merge — sum counts, union distinct values, re-sort
  // by a real column like distance_miles — can do that themselves rather
  // than having rows silently flattened together.
  return Promise.all(
    shards.map(async (s) => {
      const rows = await queryWithRetry(s.pool, sql, params);
      return { num: s.num, rows };
    })
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
