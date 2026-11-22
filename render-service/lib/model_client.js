"use strict";
// Talks to the live researchers-company-scrapper-chat worker — OUR OWN
// fine-tuned model (base Qwen2.5-7B-Instruct + the project's frozen
// "template" adapter + this worker's own trained adapter, stacked), NEVER
// a third-party LLM provider. See workers/researchers-company-scrapper-chat/
// for the training script, training data, SYSTEM_PROMPT.txt (the exact
// contract this returns), model_server.py (what's actually running), and
// modal_deploy.py (where it's actually hosted — Modal, serverless GPU).
//
// This client is intentionally "fail soft, never fail loud": if
// MODEL_SERVER_URL isn't set yet, or the model server is cold-starting,
// unreachable, or returns something malformed, understandWithModel()
// just returns null — server.js's /api/chat route then falls straight
// back to the deterministic regex pipeline (lib/chat.js's parseRequest/
// isCountQuery/isNearMeQuery/extractCountSubject), which keeps answering
// real requests the whole time this worker is being trained/deployed/
// redeployed. The user never sees "the AI is down" — they just
// transparently get the slightly-less-flexible deterministic parser
// until the model is back.

const MODEL_SERVER_URL = process.env.MODEL_SERVER_URL;
const TIMEOUT_MS = Number(process.env.MODEL_SERVER_TIMEOUT_MS || 8000);

const _VALID_INTENTS = new Set(["search", "count", "clarify", "unsupported"]);
const _VALID_MODES = new Set(["chat", "csv", "pdf"]);

function isWellFormed(data) {
  return (
    data && data.ok === true &&
    _VALID_INTENTS.has(data.intent) &&
    _VALID_MODES.has(data.delivery_mode) &&
    (data.category_phrase === null || typeof data.category_phrase === "string") &&
    (data.address === null || typeof data.address === "string") &&
    (data.radius_miles === null || typeof data.radius_miles === "number") &&
    typeof data.no_website_only === "boolean"
  );
}

/** Returns the worker model's structured intent object, or null if the
 * model isn't configured/reachable/well-formed — in which case the
 * caller MUST fall back to the deterministic parser rather than error
 * out. Never throws. */
async function understandWithModel(message, context) {
  if (!MODEL_SERVER_URL) return null;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const resp = await fetch(`${MODEL_SERVER_URL.replace(/\/+$/, "")}/understand`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, context: context || null }),
      signal: controller.signal,
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    if (!isWellFormed(data)) return null;
    return data;
  } catch (e) {
    return null; // timeout, cold start, DNS failure, bad JSON, etc. — all treated the same: fall back.
  } finally {
    clearTimeout(timer);
  }
}

module.exports = { understandWithModel };
