"use strict";
// Single shared Supabase client, using the SERVICE ROLE key — this process
// is the only thing with write access to the businesses table (the Kaggle
// scanner never gets Supabase credentials at all; it posts batches to this
// service's /api/ingest/businesses instead, which is the whole point of the
// split: Supabase creds live in exactly one place, Render).
const { createClient } = require("@supabase/supabase-js");

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  throw new Error(
    "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set as Render environment " +
    "variables before this service can start. SUPABASE_SERVICE_KEY is the " +
    "SECRET (service_role) key from Supabase Project Settings -> API — never " +
    "the publishable/anon key, which can't write."
  );
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false },
});

module.exports = { supabase };
