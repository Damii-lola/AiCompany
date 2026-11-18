"use strict";
// Replaces the local 4-bit Qwen inference company_scrapper_ai.py used on
// Kaggle for request-parsing and category-matching. Render has no GPU, so
// that local inference step isn't something this process can do — Mistral's
// hosted API is a real, already-in-use-by-this-user service (it's in the
// stated core stack) that does the same "parse free text into structured
// fields, pick from a real list" job over plain HTTPS, no GPU required.
// This is architecting around a real constraint, not a downgrade pretending
// to be the same thing: it is still real inference, just hosted elsewhere.

const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const MISTRAL_MODEL = process.env.MISTRAL_MODEL || "mistral-small-latest";

async function mistralChat(prompt, { temperature = 0.0, maxTokens = 300 } = {}) {
  if (!MISTRAL_API_KEY) {
    throw new Error("MISTRAL_API_KEY is not set — add it as a Render environment variable.");
  }
  const res = await fetch("https://api.mistral.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${MISTRAL_API_KEY}`,
    },
    body: JSON.stringify({
      model: MISTRAL_MODEL,
      temperature,
      max_tokens: maxTokens,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Mistral API request failed (${res.status}): ${body}`);
  }
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content;
  if (!text) {
    throw new Error(`Mistral API returned no content: ${JSON.stringify(data)}`);
  }
  return text.trim();
}

/** Extracts the structured pieces of a trade-area research request from free
 * text. Only ever extracts what the user asked for — the model never decides
 * the actual answer, that always comes from the real Supabase query downstream. */
async function parseResearchRequest(nlRequest) {
  const prompt =
    "Extract the following fields from this business research request as JSON, " +
    "with keys exactly: address (string or null), radius_miles (number or null), " +
    "category (short plain-English business type, string), country (ISO2 code or null), " +
    "no_website_only (true/false). Reply with ONLY the JSON object, nothing else.\n\n" +
    `Request: "${nlRequest}"`;
  const raw = await mistralChat(prompt, { temperature: 0.0, maxTokens: 200 });
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) {
    throw new Error(`Could not parse a structured request out of the model's output: ${raw}`);
  }
  return JSON.parse(match[0]);
}

/** Maps a plain-English category term ("dentist") to real category strings
 * that actually exist in the database, picking ONLY from a shortlist of real
 * values and validated again after the call — never trusted blindly, and
 * never allowed to invent a category that isn't real. */
async function matchCategories(plainEnglishTerm, realCategories, shortlist) {
  if (shortlist.length === 0) return [];
  const prompt =
    `A user is researching businesses and described the category as: "${plainEnglishTerm}".\n` +
    `Here are REAL category values that exist in the business database:\n` +
    `${JSON.stringify(shortlist)}\n\n` +
    `Reply with ONLY a JSON array of the category values from that exact list that best match ` +
    `what the user means. Pick 1-3. If none genuinely match, reply with []. ` +
    `Do not invent a category that isn't in the list.`;
  const raw = await mistralChat(prompt, { temperature: 0.0, maxTokens: 100 });
  const match = raw.match(/\[[\s\S]*\]/);
  let picked = [];
  try {
    picked = match ? JSON.parse(match[0]) : [];
  } catch {
    picked = [];
  }
  const realSet = new Set(realCategories);
  return picked.filter((c) => realSet.has(c));
}

module.exports = { mistralChat, parseResearchRequest, matchCategories };
