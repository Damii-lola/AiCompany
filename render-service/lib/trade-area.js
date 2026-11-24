"use strict";
// Radius ("trade area") search — TiDB has no stored functions/procedures
// and no spatial types, so this replaces the two Postgres SQL functions
// businesses_within_radius()/businesses_trade_area_summary() (cube +
// earthdistance) that used to live in the Postgres schema. The real
// distance math moves here instead: each shard is asked for a cheap
// bounding-box pre-filter (a plain lat/lon range, using the
// idx_businesses_latlon index from database/schema.mysql.sql), then this
// file computes the exact real great-circle (haversine) distance for every
// candidate row in JS and keeps only the ones genuinely inside the radius.
// This is the same real spherical-earth distance math the Postgres
// extension used (not an approximation of it) — just computed here instead
// of inside the database.

const { queryEachShard } = require("./db");

const EARTH_RADIUS_MILES = 3958.8;
const MILES_PER_DEGREE_LAT = 69.172; // standard constant, same scale earthdistance used

function toRad(deg) {
  return (deg * Math.PI) / 180;
}

/** Real great-circle distance between two lat/lon points, in miles. */
function haversineMiles(lat1, lon1, lat2, lon2) {
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return EARTH_RADIUS_MILES * c;
}

/** A rectangular lat/lon window that's a guaranteed SUPERSET of the real
 * circle of the given radius around (centerLat, centerLon) — cheap for the
 * database to filter on with a plain index, never excludes a true match.
 * The exact circle is enforced afterward in JS via haversineMiles. Clamps
 * the latitude term away from 0 near the poles (never relevant for real US
 * addresses, but avoids a divide-by-huge-number edge case). */
function boundingBox(centerLat, centerLon, radiusMiles) {
  const deltaLat = radiusMiles / MILES_PER_DEGREE_LAT;
  const cosLat = Math.max(Math.cos(toRad(centerLat)), 0.01);
  const deltaLon = radiusMiles / (MILES_PER_DEGREE_LAT * cosLat);
  return {
    minLat: centerLat - deltaLat,
    maxLat: centerLat + deltaLat,
    minLon: centerLon - deltaLon,
    maxLon: centerLon + deltaLon,
  };
}

async function tradeAreaFanOut(lat, lon, radiusMiles, categories, noWebsiteOnly, country) {
  const { minLat, maxLat, minLon, maxLon } = boundingBox(lat, lon, radiusMiles);

  const clauses = [
    "lat IS NOT NULL", "lon IS NOT NULL",
    "lat BETWEEN ? AND ?", "lon BETWEEN ? AND ?",
  ];
  const params = [minLat, maxLat, minLon, maxLon];
  if (categories && categories.length) {
    clauses.push("category IN (?)");
    params.push(categories);
  }
  if (noWebsiteOnly) {
    clauses.push("has_website = 0");
  }
  if (country) {
    clauses.push("country = ?");
    params.push(country);
  }

  const perShard = await queryEachShard(
    `SELECT id, name, category, category_extra, address, city, region, country, postcode,
            phone, website, email, social, brand, confidence, operating_status, lon, lat,
            first_seen_at, last_seen_at
     FROM businesses_readable
     WHERE ${clauses.join(" AND ")}`,
    params
  );

  // The bounding box is a rectangle, not a circle, so it can include real
  // rows outside the true radius (e.g. a far corner of the box) — this is
  // where those get filtered out for real, using the actual distance.
  const businesses = perShard
    .flatMap((s) => s.rows)
    .map((r) => ({ ...r, distance_miles: haversineMiles(lat, lon, r.lat, r.lon) }))
    .filter((r) => r.distance_miles <= radiusMiles)
    .sort((a, b) => a.distance_miles - b.distance_miles);

  const competitor_count = businesses.length;
  const no_website_count = businesses.filter((r) => !r.website || !String(r.website).trim()).length;
  const nearest_miles = businesses.length ? businesses[0].distance_miles : null;

  return { businesses, summary: { competitor_count, no_website_count, nearest_miles } };
}

module.exports = { tradeAreaFanOut, haversineMiles, boundingBox };
