"use strict";
// Single shared Postgres pool, talking directly to Prisma Postgres —
// replaces lib/supabase.js. Prisma Postgres has no PostgREST-style REST
// API like Supabase did, so this service now speaks plain SQL over the
// `pg` driver instead of the @supabase/supabase-js client. Nothing about
// the credential-isolation rule changes: this is still the only process
// holding the database connection string (the scanner holds its own
// copy as a separate GitHub Actions secret; the Kaggle fallback scanner
// still never gets one — it posts to /api/ingest/businesses instead).
const { Pool } = require("pg");

const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  throw new Error(
    "DATABASE_URL must be set as a Render environment variable before this " +
    "service can start — the Prisma Postgres connection string from your " +
    "Prisma dashboard (Connect -> the postgres:// string starting with " +
    "postgres://...@db.prisma.io:5432/postgres?sslmode=require, or the " +
    "pooled.db.prisma.io variant)."
  );
}

const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: { rejectUnauthorized: false }, // Prisma Postgres requires sslmode=require
});

async function query(sql, params = []) {
  const { rows } = await pool.query(sql, params);
  return rows;
}

module.exports = { pool, query };
