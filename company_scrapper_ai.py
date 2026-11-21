"""
Researchers / Company Scrapper — AI-assisted competitor & business discovery
==============================================================================
First AI in the "Researchers" category (Company Scrapper, Basic Researcher,
Advance Researcher, Organizer). Its job: given a plain-English research
request, find the real companies that match — for SBA feasibility
pre-screening specifically (trade-area competitor counts, no-website
breakdown), but built generally enough that Basic/Advance Researcher can
reuse the same tool layer later.

This is the "addon to the template AI" that turns a plain-English request
into a real query against the live database: a user types what they want,
this worker (base Qwen + the fresh template adapter, no worker adapter of
its own — see below for why) extracts the structured pieces of the
request, picks real category matches from the REAL live category list
(never invents one), and hands back real files — CSV/XLSX/PDF — built from
actual rows, never a model-generated guess at what the data contains.

WHY NO WORKER ADAPTER (explicit judgment call, per the standing allowance
"this is something that might not need finetuning but if u feel it does,
then we can finetune it"):
This AI's actual job is turning a loose request into exact calls against
REAL deterministic tools — geocoding, radius math, and matching plain-English
category words to the REAL category strings the live database actually
uses (e.g. a user says "dentist", the database only ever has
"dental clinic"). Every one of those steps has a verifiable right answer
that already lives in code or in the real distinct-category list pulled
live from the database — none of it is a question of "sound more like an
SBA analyst," which is what fine-tuning would actually buy. Baking
"dentist -> dental clinic" into model weights would be strictly worse than
just handing the model the real list and having it pick from THAT, because
the list can grow or change and weights can't update themselves. So this
script is inference + real tool-calling on the CURRENT template (pulled
fresh every run, same as every worker), with NO training loop and no
adapter of its own. If real use later shows the model consistently
mis-parsing a particular kind of SBA request in a way that more tool code
can't fix (not "needs nicer phrasing" but "needs judgment"), THAT's the
moment to come back and give it a worker_fine_tune.py pass with real
labeled (request -> correct spec) examples via the fork_worker_ai.py
pattern — not before, and not speculatively.

WHAT THIS INTEGRATES:
  1. render-service's live HTTP API (aicompany-tmg7.onrender.com) — the
     real, continuously-correct source of truth: ~18.8M real US+territory
     businesses, sharded across multiple Prisma Postgres databases, kept
     current by scanner-service/scan_worker.py on its own GitHub Actions
     schedule. THIS script has zero responsibility for scanning or
     freshness anymore, it only ever reads. (Superseded Nov 2026: this
     used to pull a Kaggle-Dataset parquet snapshot and track its own
     "is a rescan due" state against Overture's STAC release feed — that
     whole architecture is gone along with Kaggle-as-the-scanner. Live
     Postgres behind an API is always current; there is no snapshot here
     to go stale and nothing left for this script to track.)
  2. service_sba_trade_area_snapshot.py — thin client over that same live
     API: real radius-based competitor queries + geocoding.
  3. service_lead_research_report.py — the exact-match (non-radius) query
     path, for requests that aren't trade-area shaped.
  4. The CURRENT template model (pulled fresh from the GitHub repo, same
     as every worker) — parses the free-text request and picks real
     category matches from the real, live category list, nothing invented.
"""

# %% Install
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
get_ipython().system('pip install -q -U transformers accelerate "bitsandbytes>=0.46.1" peft requests')

# %% Config
import json, time, shutil, re, subprocess, difflib
import torch

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Namespace for this AI's own bits (none persisted right now — see docstring:
# this worker has no state of its own left to track since freshness/scanning
# moved entirely to scan_worker.py).
WORKER_SLUG = "researchers-company-scrapper"

GITHUB_REPO_SLUG = "Damii-lola/AiCompany"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO_SLUG}"
GITHUB_USERNAME = "Damii-lola"
LOCAL_REPO_DIR = "/kaggle/working/_repo"
TEMPLATE_REPO_PATH = "model"
TEMPLATE_DIR = "/kaggle/working/_template_adapter"

# The live read API (render-service) — same URL the GitHub Pages dashboard
# and service_sba_trade_area_snapshot.py call. Sharded behind the scenes
# (render-service/lib/db.js fans out across every configured
# DATABASE_URL_N and merges), none of which this script needs to know
# about — it's one URL from here.
RENDER_API_BASE = "https://aicompany-tmg7.onrender.com"

# %% GitHub permanent storage — same git helpers every other script here uses
from kaggle_secrets import UserSecretsClient
_s = UserSecretsClient()


def _run_git(cmd, cwd=None, check=True):
    printable = " ".join(cmd)
    if "@github.com" in printable:
        printable = re.sub(r"https://[^@]+@github\.com", "https://***@github.com", printable)
    print(f"  $ {printable}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0 and check:
        print(result.stderr.strip())
        raise RuntimeError(f"git command failed: {printable}\n{result.stderr}")
    return result


def clone_repo_fresh():
    if os.path.exists(LOCAL_REPO_DIR):
        shutil.rmtree(LOCAL_REPO_DIR)
    _run_git(["git", "clone", GITHUB_REPO_URL, LOCAL_REPO_DIR])
    return LOCAL_REPO_DIR


repo_dir = clone_repo_fresh()

# %% Pull the template — ALWAYS fresh, every run, read-only here. This AI
# reasons with whatever the template's current best state is, automatically.
template_src = os.path.join(repo_dir, TEMPLATE_REPO_PATH)
assert os.path.exists(os.path.join(template_src, "adapter_model.safetensors")), (
    f"No template adapter found at {TEMPLATE_REPO_PATH}/ in {GITHUB_REPO_SLUG} — "
    f"train the template with train_the_ai.py first."
)
if os.path.exists(TEMPLATE_DIR):
    shutil.rmtree(TEMPLATE_DIR)
shutil.copytree(template_src, TEMPLATE_DIR)
with open(os.path.join(TEMPLATE_DIR, "training_progress.json")) as f:
    template_steps = json.load(f).get("total_steps")
print(f"Pulled the CURRENT template — cumulative {template_steps:,} steps as of right now.")

# %% Load base model in 4-bit + the current template adapter (inference only
# — no training, no worker adapter; see docstring for why)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nLoading base model in 4-bit + current template (inference only)...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, TEMPLATE_DIR, adapter_name="template", is_trainable=False)
model.set_adapter("template")
model.eval()


def generate(prompt, max_new_tokens=400, temperature=0.2):
    """temperature=0.2 default on purpose — this is used for structured
    extraction (parsing a request into a spec), where we want the model's
    most confident reading, not creative variety."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                              temperature=max(temperature, 1e-5), top_p=0.9)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

# %% Real category taxonomy — pulled LIVE from the actual live database via
# render-service's API, never invented and never a stale local snapshot.
# This is what makes "search so much better" real: the model picks from
# what the data actually contains right now, instead of guessing a
# category string that sounds right but doesn't exist (which would
# silently return zero rows and look like a bug).
import requests

_cat_resp = requests.get(f"{RENDER_API_BASE}/api/categories", timeout=30)
_cat_resp.raise_for_status()
REAL_CATEGORIES = sorted(_cat_resp.json()["categories"])
print(f"Loaded {len(REAL_CATEGORIES):,} real distinct category values from the live database.")


def match_categories(plain_english_term, top_k=5):
    """Maps what a user actually says ('dentist', 'coffee shop') to the
    REAL category strings in the database. Two-stage, both grounded in the
    real list — never a free-form guess:
      1. difflib close-match against the real list (fast, catches obvious
         matches like 'dentist' -> 'dental clinic')
      2. the model picks from a shortlist combining close matches + a
         sample of the full list, forced to answer ONLY with values that
         are actually in REAL_CATEGORIES (checked after, not trusted blindly)
    Returns a list of real category strings, never a fabricated one."""
    term = plain_english_term.strip().lower()
    close = difflib.get_close_matches(term, REAL_CATEGORIES, n=top_k, cutoff=0.3)
    # Overture's real taxonomy values are typically snake_case ("coffee_shop",
    # "dental_clinic"), but this is matched defensively against BOTH that and
    # a plain space-separated form — rather than assume one convention, since
    # getting this wrong silently misses real matches instead of erroring.
    term_us, term_sp = term.replace(" ", "_"), term.replace("_", " ")
    token_matches = [
        c for c in REAL_CATEGORIES
        if term in c or term_us in c or term_sp in c or c in term or c.replace("_", " ") in term
    ]
    shortlist = list(dict.fromkeys(close + token_matches))[:15]

    if not shortlist:
        return []

    prompt = (
        f"A user is researching businesses and described the category as: \"{plain_english_term}\".\n"
        f"Here are REAL category values that exist in the business database:\n"
        f"{json.dumps(shortlist)}\n\n"
        f"Reply with ONLY a JSON array of the category values from that exact list that best match "
        f"what the user means. Pick 1-3. If none genuinely match, reply with []. "
        f"Do not invent a category that isn't in the list."
    )
    raw = generate(prompt, max_new_tokens=100, temperature=0.0)
    try:
        picked = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
    except Exception:
        picked = []
    # Never trust the model's output blindly — validate against the real set.
    valid = [c for c in picked if c in REAL_CATEGORIES]
    return valid or close[:top_k]

# %% Request parsing — turns a plain-English research request into a real
# structured spec for service_sba_trade_area_snapshot.fulfil()
import service_sba_trade_area_snapshot as sba


def parse_request(nl_request: str) -> dict:
    """Extracts the structured pieces of an SBA trade-area request from
    free text. The model's job here is ONLY extraction (what did the user
    ask for), never deciding the answer — the actual competitor numbers
    always come from the real live database downstream, not from the
    model."""
    prompt = (
        "Extract the following fields from this business research request as JSON, "
        "with keys exactly: address (string or null), radius_miles (number or null), "
        "category (short plain-English business type, string), country (ISO2 code or null), "
        "no_website_only (true/false). Reply with ONLY the JSON object, nothing else.\n\n"
        f"Request: \"{nl_request}\""
    )
    raw = generate(prompt, max_new_tokens=200, temperature=0.0)
    try:
        parsed = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception as e:
        raise ValueError(f"Could not parse a structured request out of the model's output: {raw!r} ({e})")
    return parsed


def research(nl_request: str, output_formats=None, out_dir="./company_scrapper_out") -> dict:
    """The main entry point: plain-English request in, real results out.
    Every field in the returned spec either came from the user's own words
    or was validated against real data (real geocode, real category match,
    real row counts) — nothing here is invented to fill a gap.

    Geocodes + queries by EXACT coordinates/category (service_sba_trade_area
    _snapshot's form (b)) rather than letting render-service's own fuzzy
    matching run a second time — this worker's whole value-add over a bare
    API call is doing that category matching WITH the model's judgment
    (match_categories above), so by the time we call the live API we
    already have the real, validated category string(s)."""
    parsed = parse_request(nl_request)
    print(f"Parsed request: {parsed}")

    if not parsed.get("address"):
        raise ValueError("Request has no address/location to anchor the trade area — "
                          "ask the user for one rather than guessing a location.")
    lat, lon = sba.geocode_address(parsed["address"])
    print(f"Geocoded '{parsed['address']}' -> ({lat:.4f}, {lon:.4f})")

    categories = match_categories(parsed.get("category", ""))
    if not categories:
        raise ValueError(f"Could not match '{parsed.get('category')}' to any real category in the "
                          f"database — ask the user to clarify rather than scanning everything.")
    print(f"Matched category '{parsed.get('category')}' -> real categories: {categories}")

    spec = {
        "center_lat": lat, "center_lon": lon,
        "radius_miles": parsed.get("radius_miles") or 3,
        "category": categories,
        "country": parsed.get("country"),
        "no_website_only": bool(parsed.get("no_website_only")),
        "output_formats": output_formats or ["csv", "pdf"],
        "api_base": RENDER_API_BASE,
    }
    preview = sba.preview_trade_area(spec)
    print(f"Preview: {preview}")
    files = sba.fulfil(spec, out_dir)
    return {"spec": spec, "preview": preview, "files": files}


print(f"\nCompany Scrapper ready. template@{template_steps:,}, "
      f"{len(REAL_CATEGORIES):,} real categories loaded live from {RENDER_API_BASE}.\n"
      f"Call research(\"find dental clinics with no website within 3 miles of 123 Main St, "
      f"Columbus, OH\") to run a real request.")
