"use strict";
// Live chat system — the "addon to the template AI" the user asked for,
// moved OFF GitHub Actions (.github/workflows/research.yml, now deleted)
// and onto the always-on render-service backend, so docs/index.html can
// call it directly over HTTP for an instant response instead of a
// manual workflow_dispatch form + artifact download.
//
// This is a straight JS port of company_scrapper_ai.py's deterministic
// parsing (radius/no-website regex, filler/connector stripping, the
// category/address window-splitter) — NOT a reuse of the Python file,
// since this path has to run inside this Node process. Keep the two in
// sync if either changes; company_scrapper_ai.py is now superseded by
// this file for live requests (it's still fine as a standalone CLI/local
// tool, just no longer part of the live request path).
//
// Second half of the user's request: "listen for the output forms, from
// the user request, it should determine if it want just basic in chat
// response, or one of the output forms" — decideDeliveryMode() below
// scans the message itself for format-indicating words, no separate
// format selector control needed.

const { categoryScore } = require("./categories");

const _RADIUS_RE = /(?:within\s+)?(\d+(?:\.\d+)?)\s*[- ]?mi(?:les?)?\b(?:\s+radius)?(?:\s+of\b)?/i;
const _NO_WEBSITE_RE = /(?:with\s+|that\s+(?:have|has)\s+)?(?:no|without|missing)\s+(?:a\s+|an?y\s+)?website/i;
const _LEAD_FILLER = [
  "please find", "please show", "please search for", "please get me",
  "find", "search for", "get me", "show me", "look for", "i need", "i want",
];
const _CONNECTORS = ["near", "around", "in", "of", "with", "for", "at"];
// A real address has at least one of: a number (street number or zip), a
// "City, ST" comma-state pattern, or a street-suffix word. Used ONLY to
// gate the no-category-matched fallback below — without this, any
// contentless follow-up ("list all of them", "give me that as a csv")
// got its entire leftover text dumped into "address" by default and
// treated as a brand-new location to geocode, which both produced
// nonsense geocoder lookups and (worse, once conversation context was
// added) wrongly told resolveWithContext "this message names its own
// new place," discarding the real context it should have fallen back on.
const _ADDRESS_HINT_RE = /\d{2,}|,\s*[A-Za-z]{2}\b|\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|boulevard|blvd\.?|lane|ln\.?|way|highway|hwy|parkway|pkwy)\b/i;

/** Strips any of `words` repeatedly from the START of text — handles the
 * case where a connector word ("near") IS the very first word left over
 * after the category phrase is cut out, which a leading-whitespace-
 * assuming regex misses. */
function stripWords(text, words) {
  let t = text.trim();
  let changed = true;
  while (changed) {
    changed = false;
    const low = t.toLowerCase();
    for (const w of words) {
      if (low.startsWith(w + " ") || low === w) {
        t = t.slice(w.length).trim();
        changed = true;
        break;
      }
    }
  }
  return t;
}

function stripLeadingFiller(text) {
  return stripWords(text, _LEAD_FILLER);
}

/** Scans every 1-4 word window of `words` for the one that best
 * fuzzy-matches a REAL category (via categoryScore, same scoring used
 * by closeMatches), so a category phrase is always anchored to real
 * data — never an invented guess. Shared by splitCategoryAndAddress
 * (which also needs the leftover words as an address) and the count-
 * query path (which just needs the category, ignoring noise words like
 * "eligible for loans" that don't correspond to any real data field).
 *
 * A single-word window needs a real match, not a weak fuzzy brush-by —
 * at live scale (2,000+ real categories) a short, generic word like
 * "them" or "list" WILL incidentally clear a loose 0.45 bar against
 * something in that large a vocabulary purely by chance, which is
 * exactly what once turned a contentless follow-up ("list all of them")
 * into a bogus category match instead of being recognized as having no
 * real category/location content at all. */
function findCategoryWindow(words, realCategories) {
  if (words.length === 0) return null;
  const normCats = realCategories.map((c) => c.replace(/_/g, " ").toLowerCase());
  let bestScore = 0, bestSpan = null;
  for (let size = 1; size <= Math.min(4, words.length); size++) {
    for (let i = 0; i <= words.length - size; i++) {
      const window = words.slice(i, i + size).join(" ").replace(/^[\s,.]+|[\s,.]+$/g, "").toLowerCase();
      if (!window) continue;
      let bestForWindow = 0;
      for (const nc of normCats) {
        const s = categoryScore(window, nc);
        if (s > bestForWindow) bestForWindow = s;
      }
      const score = bestForWindow + size * 0.001;
      if (score > bestScore) {
        bestScore = score;
        bestSpan = [i, i + size];
      }
    }
  }
  const minScore = (bestSpan && bestSpan[1] - bestSpan[0] === 1) ? 0.55 : 0.45;
  if (!bestSpan || bestScore < minScore) return null;
  const [i0, i1] = bestSpan;
  return {
    category: words.slice(i0, i1).join(" ").replace(/^[\s,.]+|[\s,.]+$/g, ""),
    remainderWords: [...words.slice(0, i0), ...words.slice(i1)],
    leftoverWords: [...words.slice(0, i0), ...words.slice(i1)].filter(Boolean),
  };
}

function splitCategoryAndAddress(text, realCategories) {
  const words = text.replace(/^[\s,.]+|[\s,.]+$/g, "").split(/\s+/).filter(Boolean);
  if (words.length === 0) return { category: "", address: "" };

  const found = findCategoryWindow(words, realCategories);
  if (!found) {
    const full = words.join(" ").replace(/^[\s,.]+|[\s,.]+$/g, "");
    return { category: "", address: _ADDRESS_HINT_RE.test(full) ? full : "" };
  }

  let remainder = stripWords(found.remainderWords.join(" "), _CONNECTORS);
  remainder = remainder.replace(/\s{2,}/g, " ").replace(/^[\s,.]+|[\s,.]+$/g, "");
  return { category: found.category, address: remainder };
}

/** Extracts the structured pieces of a request from free text,
 * deterministically. Every field is either a direct pattern match or
 * validated against real, live category data downstream — nothing here
 * decides the final answer, it only decides what to ask for. */
function parseRequest(nlRequest, realCategories) {
  let text = nlRequest.trim();

  let radiusMiles = null;
  const m = _RADIUS_RE.exec(text);
  if (m) {
    radiusMiles = parseFloat(m[1]);
    text = (text.slice(0, m.index) + " " + text.slice(m.index + m[0].length)).trim();
  }

  let noWebsiteOnly = false;
  const m2 = _NO_WEBSITE_RE.exec(text);
  if (m2) {
    noWebsiteOnly = true;
    text = (text.slice(0, m2.index) + " " + text.slice(m2.index + m2[0].length)).trim();
  }

  text = stripLeadingFiller(text);
  text = text.replace(/\s{2,}/g, " ").replace(/^[\s,.]+|[\s,.]+$/g, "");

  const { category, address } = splitCategoryAndAddress(text, realCategories);
  return {
    address: address || null,
    radius_miles: radiusMiles,
    category,
    country: "US", // this whole platform only scans the US + SBA-lending territories
    no_website_only: noWebsiteOnly,
  };
}

/** Conversation-memory follow-up resolution — deterministic, not an LLM
 * guess: a chat is a sequence of turns, and a follow-up like "list all
 * of them" or "now show me only the ones with no website" or "what
 * about within 5 miles instead" has no location/category of its own on
 * purpose — it's referring BACK to whatever the previous turn already
 * resolved. `context` is the previous turn's resolved spec (address,
 * lat/lon, matchedCategories, radiusMiles, noWebsiteOnly), round-
 * tripped by the client every request (render-service itself holds no
 * server-side session state — the browser's own chat history is the
 * source of truth, same as this file's design elsewhere: nothing here
 * decides the final answer on a guess).
 *
 * Rule: if the new message names its OWN address, it's a fresh query —
 * context is dropped entirely (predictable: a new place always starts
 * clean). Otherwise, whatever the new message DID detect (a new
 * category, a new radius, a no-website toggle) overrides the matching
 * field in context; anything it didn't mention carries over unchanged.
 * A message with nothing new at all ("list all of them") carries over
 * everything — that's the exact case that used to crash. */
function resolveWithContext(parsed, matchedCategories, context) {
  if (parsed.address) {
    // A real new location was named — fresh query, no carryover at all.
    return {
      address: parsed.address,
      matchedCategories,
      radiusMiles: parsed.radius_miles || 3,
      noWebsiteOnly: parsed.no_website_only,
      needsGeocode: true,
    };
  }
  if (!context) return null; // nothing of its own and nothing to fall back on

  return {
    address: context.address,
    lat: context.lat,
    lon: context.lon,
    matchedCategories: matchedCategories.length ? matchedCategories : context.matchedCategories,
    radiusMiles: parsed.radius_miles || context.radiusMiles || 3,
    noWebsiteOnly: parsed.no_website_only || !!context.noWebsiteOnly,
    needsGeocode: false,
  };
}

/** Same merge semantics as resolveWithContext above, but driven by the
 * worker model's structured intent object (lib/model_client.js) instead
 * of the deterministic regex parser. Deliberately kept as a SEPARATE,
 * parallel function rather than folded into resolveWithContext: the two
 * callers pass differently-shaped inputs (a `parsed` object with a raw
 * `category` string vs. an `intentObj` with an already-typo-corrected
 * `category_phrase`), and keeping them apart means a change to one
 * understanding layer's quirks never risks the other's behavior. The
 * MERGE RULE itself is identical on purpose — "a named address starts
 * fresh, everything else carries over from context" — because that rule
 * is about how a multi-turn chat behaves, not about which layer
 * extracted the fields. */
function resolveModelIntent(intentObj, matchedCategories, context) {
  if (intentObj.address) {
    return {
      address: intentObj.address,
      matchedCategories,
      radiusMiles: intentObj.radius_miles || 3,
      noWebsiteOnly: !!intentObj.no_website_only,
      needsGeocode: true,
    };
  }
  if (!context) return null;
  return {
    address: context.address,
    lat: context.lat,
    lon: context.lon,
    matchedCategories: matchedCategories.length ? matchedCategories : context.matchedCategories,
    radiusMiles: intentObj.radius_miles || context.radiusMiles || 3,
    noWebsiteOnly: !!intentObj.no_website_only || !!context.noWebsiteOnly,
    needsGeocode: false,
  };
}

// ---------------------------------------------------------------------
// Count / aggregate questions — "how many coffee shops are in the
// database", "number of pharmacies", "count of dental clinics ur
// database has". These have NO location at all and were previously
// (wrongly) forced through the address/geocoding path, where a short,
// locationless phrase like "in ur database" either failed to parse an
// address and errored, or — worse — got misread as an address and sent
// to the geocoder, producing a confusing "no match" error for a
// question that was never about a place. Detected and answered
// separately, with a real nationwide (no radius) count query.
// ---------------------------------------------------------------------
const _COUNT_RE = /\b(?:how many|number of|count of|total (?:number|count) of)\b/i;
// Loose fallback: a message that LEADS with "number"/"count"/"total" is
// almost always a count question even when the word that should follow
// ("of") got mangled by autocorrect/swipe-typing — e.g. a real message
// seen in practice, "Number 9f bakeries near me" (keyboard turned "of"
// into "9f"). The exact phrase regex above would miss this entirely.
const _COUNT_LOOSE_RE = /^\s*(?:number|count|total)\b/i;
const _DB_FILLER_RE = /\s*(?:in|from|on|within)\s+(?:the|ur|your|our|my)?\s*(?:database|db|system|records?)\b\.?\s*$/i;
// Generic helper/subject words that carry no category meaning of their
// own ("are", "is", "businesses") — stripped wherever they appear, not
// just at the edges, since "how many businesses are..." leaves "are"
// stranded mid-phrase after the trailing db-filler is cut off the end.
const _GENERIC_SUBJECT_WORDS = /\b(?:are|is|there|do|does|we|you|have|has|business(?:es)?|compan(?:y|ies)|listed|available)\b/gi;

function isCountQuery(message) {
  return _COUNT_RE.test(message) || _COUNT_LOOSE_RE.test(message);
}

/** Does the message ask for something tied to the asker's own physical
 * location ("near me", "around me", "close to me", "by me")? This
 * platform has no device/browser geolocation at all — there's no
 * "me" to resolve — so sending that literal text to the geocoder just
 * produces a confusing "no match" error. Checked explicitly so the
 * error can say what's actually wrong instead. */
const _NEAR_ME_RE = /\b(?:near|around|close to|by)\s+me\b/i;
function isNearMeQuery(message) {
  return _NEAR_ME_RE.test(message);
}

/** Strips the count phrase itself ("how many", "number of"... or the
 * loose "number"/"count"/"total" lead-in, typo and all), any trailing
 * "...in the database" filler, leading request filler, and generic
 * helper words with no category meaning. Then runs the SAME real-
 * category window search used for trade-area requests over whatever's
 * left — so noise words that don't correspond to any real data field
 * ("eligible for loans") are correctly dropped rather than sinking the
 * whole match, same as a trade-area request tolerates noise around its
 * category+address.
 *
 * Returns one of three shapes, distinguished on purpose rather than
 * collapsed into "no category" for all of them:
 *   - { category: null } — nothing was left to say at all ("how many
 *     businesses are there") -> count EVERYTHING, not an error.
 *   - { category, ignoredWords } — a real category was found; anything
 *     left over (that didn't fit the real taxonomy) is reported, not
 *     silently applied or silently dropped without a trace.
 *   - { noMatch: true, text } — real-looking content remained but
 *     nothing in it matched any real category ("unicorns"). This must
 *     error out, not silently fall back to counting everything — that
 *     would answer a different question than the one actually asked. */
function extractCountSubject(message, realCategories) {
  const exact = _COUNT_RE.exec(message);
  let text;
  if (exact) {
    text = (message.slice(0, exact.index) + " " + message.slice(exact.index + exact[0].length)).trim();
  } else {
    // Loose match: drop the leading count word, then one short leftover
    // token right after it (almost certainly "of" or a typo of it) — a
    // real category window search below recovers gracefully even if
    // this guess happens to eat part of a real word instead.
    text = message.replace(/^\s*(?:number|count|total)\b\s*/i, "");
    text = text.replace(/^[a-z0-9]{1,3}\b\s*/i, "");
  }
  text = text.replace(_DB_FILLER_RE, "").trim();
  text = stripLeadingFiller(text);
  text = text.replace(_GENERIC_SUBJECT_WORDS, " ");
  text = text.replace(/\s{2,}/g, " ").replace(/^[\s,.?]+|[\s,.?]+$/g, "");
  if (!text) return { category: null, ignoredWords: [] };

  const words = text.split(/\s+/).filter(Boolean);
  const found = findCategoryWindow(words, realCategories);
  if (!found) return { noMatch: true, text };
  return { category: found.category, ignoredWords: found.leftoverWords };
}

// ---------------------------------------------------------------------
// Delivery-mode detection — "listen for the output forms, from the user
// request, it should determine if it want just basic in chat response,
// or one of the output forms". Keyword scan over the ORIGINAL message
// text (before any stripping), since a user asking for "a csv of..."
// or "...as a pdf" is describing the deliverable, not the search itself.
// ---------------------------------------------------------------------
const _PDF_WORDS = /\b(pdf)\b/i;
const _CSV_WORDS = /\b(csv|spreadsheet)\b/i;
const _XLSX_WORDS = /\b(xlsx|excel)\b/i;
const _FILE_WORDS = /\b(report|export|download|file|deliverable)\b/i;

function decideDeliveryMode(message) {
  if (_PDF_WORDS.test(message)) return "pdf";
  if (_XLSX_WORDS.test(message)) return "xlsx";
  if (_CSV_WORDS.test(message)) return "csv";
  if (_FILE_WORDS.test(message)) return "csv"; // generic "give me a report/export" -> CSV default
  return "chat";
}

// ---------------------------------------------------------------------
// Chat reply — a natural-language summary built ONLY from real numbers
// already computed by tradeAreaFanOut, never a fabricated estimate.
// ---------------------------------------------------------------------
function pluralizeLabel(label, count) {
  if (count === 1 || label.endsWith("s")) return label;
  return label.replace(/y\b/, "ies").replace(/(shop|clinic|store|service|office|center)$/, "$1s")
    || label + "s";
}

function buildChatReply({ address, matchedCategories, radiusMiles, summary, businesses, noWebsiteOnly }) {
  const rawLabel = matchedCategories.map((c) => c.replace(/_/g, " ")).join(", ");
  const catLabel = pluralizeLabel(rawLabel, summary.competitor_count);
  const lines = [];
  lines.push(
    `Found ${summary.competitor_count.toLocaleString()} ${catLabel} ` +
    `within ${radiusMiles} mile${radiusMiles === 1 ? "" : "s"} of ${address}.`
  );
  if (summary.nearest_miles != null) {
    lines.push(`Nearest one is ${Number(summary.nearest_miles).toFixed(2)} miles away.`);
  }
  if (noWebsiteOnly) {
    lines.push(`All ${summary.no_website_count.toLocaleString()} shown have no website on file.`);
  } else {
    lines.push(`${summary.no_website_count.toLocaleString()} of them have no website on file.`);
  }
  if (businesses.length > 0) {
    const top = businesses.slice(0, 10).map((b, i) =>
      `${i + 1}. ${b.name || "(unnamed)"} — ${Number(b.distance_miles).toFixed(2)} mi` +
      (b.website ? "" : " — no website")
    );
    lines.push("", "Closest matches:", ...top);
    if (businesses.length > 10) lines.push(`...and ${businesses.length - 10} more.`);
  }
  return lines.join("\n");
}

/** Natural-language reply for a count/aggregate question — built ONLY
 * from real numbers already computed by the caller's nationwide count
 * query, same "nothing fabricated" rule as buildChatReply. */
function buildCountReply({ matchedCategories, total, noWebsite, ignoredWords }) {
  const catLabel = matchedCategories && matchedCategories.length
    ? pluralizeLabel(matchedCategories.map((c) => c.replace(/_/g, " ")).join(", "), total)
    : "businesses";
  const verb = total === 1 ? "is" : "are";
  const lines = [`There ${verb} ${total.toLocaleString()} ${catLabel} in the database.`];
  if (total > 0) {
    lines.push(`${noWebsite.toLocaleString()} of them have no website on file.`);
  }
  if (ignoredWords && ignoredWords.length) {
    lines.push(
      `(Note: this only counts by category — I don't have a way to filter by "${ignoredWords.join(" ")}", `
      + `so that part of your question wasn't applied.)`
    );
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------
// CSV — plain, dependency-free builder (real fields only, same shape as
// businesses_readable so this matches what service_sba_trade_area_snapshot.py
// already produces).
// ---------------------------------------------------------------------
function csvEscape(v) {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function buildCsv(businesses) {
  const cols = ["name", "category", "address", "city", "region", "country",
    "website", "phone", "distance_miles"];
  const header = cols.join(",");
  const rows = businesses.map((b) => cols.map((c) => csvEscape(b[c])).join(","));
  return [header, ...rows].join("\n") + "\n";
}

// ---------------------------------------------------------------------
// PDF — streamed directly as the Express response via pdfkit, no temp
// file and no Python/reportlab dependency needed in this process.
// ---------------------------------------------------------------------
function streamPdf(res, { address, matchedCategories, radiusMiles, summary, businesses, noWebsiteOnly }) {
  const PDFDocument = require("pdfkit");
  const doc = new PDFDocument({ margin: 50 });
  res.setHeader("Content-Type", "application/pdf");
  res.setHeader("Content-Disposition", 'attachment; filename="trade_area_snapshot.pdf"');
  doc.pipe(res);

  const catLabel = matchedCategories.map((c) => c.replace(/_/g, " ")).join(", ");
  doc.fontSize(18).text("Trade Area Snapshot", { align: "left" });
  doc.moveDown(0.5);
  doc.fontSize(11).fillColor("#333");
  doc.text(`Location: ${address}`);
  doc.text(`Category: ${catLabel}`);
  doc.text(`Radius: ${radiusMiles} mile${radiusMiles === 1 ? "" : "s"}`);
  if (noWebsiteOnly) doc.text("Filter: no website only");
  doc.moveDown();

  doc.fontSize(13).fillColor("#000").text("Summary");
  doc.fontSize(11).fillColor("#333");
  doc.text(`Competitors found: ${summary.competitor_count.toLocaleString()}`);
  doc.text(`Without a website: ${summary.no_website_count.toLocaleString()}`);
  if (summary.nearest_miles != null) {
    doc.text(`Nearest: ${Number(summary.nearest_miles).toFixed(2)} miles`);
  }
  doc.moveDown();

  doc.fontSize(13).fillColor("#000").text("Businesses");
  doc.fontSize(9).fillColor("#333");
  for (const b of businesses.slice(0, 200)) {
    const website = b.website ? b.website : "no website";
    doc.text(
      `${b.name || "(unnamed)"} — ${Number(b.distance_miles).toFixed(2)} mi — ${website}`
    );
  }
  if (businesses.length > 200) {
    doc.moveDown(0.5).text(`...and ${businesses.length - 200} more (see CSV for the full list).`);
  }

  doc.end();
}

module.exports = {
  parseRequest,
  resolveWithContext,
  resolveModelIntent,
  decideDeliveryMode,
  buildChatReply,
  buildCsv,
  streamPdf,
  isCountQuery,
  isNearMeQuery,
  extractCountSubject,
  buildCountReply,
  // exported for testing
  stripWords,
  splitCategoryAndAddress,
};
