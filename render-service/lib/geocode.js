"use strict";
// Same real geocoder as service_sba_trade_area_snapshot.py's
// geocode_address() — OpenStreetMap Nominatim, no API key, best-effort.
// Throws on no match rather than returning a guessed coordinate.

async function geocodeAddress(address, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    const params = new URLSearchParams({ q: address, format: "json", limit: "1" });
    res = await fetch(`https://nominatim.openstreetmap.org/search?${params}`, {
      headers: { "User-Agent": "company-scrapper-service/1.0 (internal tool)" },
      signal: controller.signal,
    });
  } catch (e) {
    throw new Error(`Geocoding request failed for "${address}": ${e.message}`);
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    throw new Error(`Geocoding request failed for "${address}": HTTP ${res.status}`);
  }
  const results = await res.json();
  if (!results.length) {
    throw new Error(`Geocoding found no match for "${address}" — get coordinates another way, don't guess them.`);
  }
  return { lat: parseFloat(results[0].lat), lon: parseFloat(results[0].lon) };
}

module.exports = { geocodeAddress };
