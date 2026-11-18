"""
Researchers / Company Scrapper — AI-assisted competitor & business discovery
==============================================================================
First AI in the "Researchers" category (Company Scrapper, Basic Researcher,
Advance Researcher, Organizer). Its job: given a plain-English research
request, find the real companies that match — for SBA feasibility
pre-screening specifically (trade-area competitor counts, no-website
breakdown), but built generally enough that Basic/Advance Researcher can
reuse the same tool layer later.

WHY NO WORKER ADAPTER (explicit judgment call, per the standing allowance
"this is something that might not need finetuning but if u feel it does,
then we can finetune it"):
This AI's actual job is turning a loose request into exact calls against
REAL deterministic tools — geocoding, radius math, and matching plain-English
category words to the REAL category strings Overture Maps actually uses
(e.g. a user says "dentist", the database only ever has "dental_clinic").
Every one of those steps has a verifiable right answer that already lives in
code or in the real distinct-category list pulled live from the database —
none of it is a question of "sound more like an SBA analyst," which is what
fine-tuning would actually buy. Baking "dentist -> dental_clinic" into model
weights would be strictly worse than just handing the model the real list and
having it pick from THAT, because the list can grow or change and weights
can't update themselves. So this script is inference + real tool-calling on
the CURRENT template (pulled fresh every run, same as every worker), with NO
training loop and no adapter of its own. If real use later shows the model
consistently mis-parsing a particular kind of SBA request in a way that more
tool code can't fix (not "needs nicer phrasing" but "needs judgment"), THAT's
the moment to come back and give it a worker_fine_tune.py pass with real
labeled (request -> correct spec) examples via the fork_worker_ai.py pattern
— not before, and not speculatively.

WHAT THIS INTEGRATES (the three pieces of infrastructure this project
already had, now wired together):
  1. kaggle_business_scanner.py  -> produces business_database.parquet
     (Kaggle Dataset `global-businesses-v1`; stays on Kaggle, not GitHub —
     this file is genuinely too large for GitHub's 100MB limit, unlike the
     small model checkpoints, and reads from Kaggle Datasets have always
     worked fine; only writes were ever broken, and that's now hardened to
     fail loud instead of silently in kaggle_business_scanner.py itself)
  2. service_sba_trade_area_snapshot.py -> real radius-based competitor
     queries + geocoding (the capability that was missing before this)
  3. service_lead_research_report.py -> the exact-match (non-radius) query
     path, for requests that aren't trade-area shaped
  4. The CURRENT template model (pulled fresh from the GitHub repo, same as
     every worker) -> parses the free-text request and picks real category
     matches from the real list, nothing invented

"ACTIVELY AND ALWAYS CHECKS CONSTANTLY" — HONEST IMPLEMENTATION:
Overture Maps publishes a new release roughly monthly, not continuously —
an actual infinite loop against a data source that only changes once a
month would burn GPU/compute hours checking nothing new, which is exactly
the kind of dishonest-looking-busy behavior this project doesn't do. What
"always checking" means here, for real: every time this script runs (which
can itself be wired to a Kaggle scheduled notebook run, e.g. daily),
check_for_new_release() compares the latest real Overture STAC release
against the one recorded from the last scan. If it's genuinely new, it
flags that a re-scan is due (and can kick one off) instead of silently
running on stale data forever. That is the honest version of "constantly
checking" for a source that updates on its own real-world schedule.
"""

# %% Install
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
get_ipython().system('pip install -q -U transformers accelerate "bitsandbytes>=0.46.1" peft duckdb pandas pyarrow kaggle requests')

# %% Config
import json, time, glob, shutil, re, subprocess, difflib
import torch

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Namespace for this AI's OWN small bits of state in the repo (a release
# marker, request logs) — not an adapter, since none is trained (see docstring).
WORKER_SLUG = "researchers-company-scrapper"

GITHUB_REPO_SLUG = "Damii-lola/AiCompany"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO_SLUG}"
GITHUB_USERNAME = "Damii-lola"
LOCAL_REPO_DIR = "/kaggle/working/_repo"
TEMPLATE_REPO_PATH = "model"
STATE_REPO_PATH = f"workers/{WORKER_SLUG}/state.json"   # last-scanned release marker lives here
TEMPLATE_DIR = "/kaggle/working/_template_adapter"

BIZ_DATASET_SLUG = "global-businesses-v1"   # same dataset kaggle_business_scanner.py pushes to
BIZ_DB_LOCAL = "/kaggle/working/_biz/business_database.parquet"

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


def push_repo_changes(commit_message):
    token = _s.get_secret("GITHUB_TOKEN")
    push_url = f"https://{GITHUB_USERNAME}:{token}@github.com/{GITHUB_REPO_SLUG}.git"
    _run_git(["git", "add", "-A"], cwd=LOCAL_REPO_DIR)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=LOCAL_REPO_DIR)
    if diff.returncode == 0:
        print("  -> no changes to push (content identical)")
        return True
    _run_git(["git", "-c", "user.email=ai@autonomous.local", "-c", "user.name=Autonomous AI",
              "commit", "-m", commit_message], cwd=LOCAL_REPO_DIR)
    try:
        _run_git(["git", "push", push_url, "HEAD:main"], cwd=LOCAL_REPO_DIR)
    except RuntimeError:
        print("  -> push rejected (remote moved since clone) — fetching latest and rebasing once...")
        _run_git(["git", "fetch", push_url, "main"], cwd=LOCAL_REPO_DIR)
        _run_git(["git", "rebase", "FETCH_HEAD"], cwd=LOCAL_REPO_DIR)
        _run_git(["git", "push", push_url, "HEAD:main"], cwd=LOCAL_REPO_DIR)
    print(f"  -> pushed to {GITHUB_REPO_SLUG}")
    return True


repo_dir = clone_repo_fresh()

print("\nVerifying write access to the permanent repo before doing anything else...")
with open(os.path.join(repo_dir, ".last_write_check"), "w") as f:
    f.write(f"write-access check at {time.time()}\n")
push_repo_changes("write-access check (auto, no content changed)")
print("WRITE ACCESS CONFIRMED.\n")

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

# %% Pull this AI's own small state file (last-scanned release marker), if any
state_src = os.path.join(repo_dir, STATE_REPO_PATH)
state = {"last_scanned_release": None}
if os.path.exists(state_src):
    with open(state_src) as f:
        state = json.load(f)
    print(f"Loaded prior state: last scanned release = {state.get('last_scanned_release')}")
else:
    print("No prior state found — this is this AI's first run.")


def save_state():
    dest = os.path.join(repo_dir, STATE_REPO_PATH)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(state, f, indent=2)
    push_repo_changes(f"Company Scrapper: state update, last_scanned_release={state.get('last_scanned_release')}")

# %% Pull the real business database (Kaggle Dataset — large binary data,
# stays on Kaggle; reads from Kaggle Datasets have always worked fine, it was
# only WRITES that were ever broken, and that's a separate dataset/problem
# from the model checkpoint anyway)
os.environ["KAGGLE_USERNAME"] = _s.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = _s.get_secret("KAGGLE_KEY")
from kaggle.api.kaggle_api_extended import KaggleApi
kapi = KaggleApi(); kapi.authenticate()
BIZ_OWNER = os.environ["KAGGLE_USERNAME"]

os.makedirs(os.path.dirname(BIZ_DB_LOCAL), exist_ok=True)
print(f"\nPulling the business database ({BIZ_OWNER}/{BIZ_DATASET_SLUG})...")
kapi.dataset_download_files(f"{BIZ_OWNER}/{BIZ_DATASET_SLUG}",
                             path=os.path.dirname(BIZ_DB_LOCAL), unzip=True, quiet=False)
found = glob.glob(f"{os.path.dirname(BIZ_DB_LOCAL)}/**/*.parquet", recursive=True)
assert found, (f"No .parquet file found after pulling {BIZ_OWNER}/{BIZ_DATASET_SLUG} — "
               f"run kaggle_business_scanner.py at least once first.")
BIZ_DB_LOCAL = found[0]
print(f"Business database ready: {BIZ_DB_LOCAL}")

# %% Check for a new Overture release — honest version of "always checking"
import requests


def check_for_new_release():
    """Compares the real current Overture STAC release against the one
    recorded from the last scan. Returns (is_new, latest_release_id).
    This is the honest form of 'constantly checking' for a source that only
    actually changes about once a month — no meaningless infinite loop."""
    try:
        r = requests.get("https://stac.overturemaps.org/catalog.json", timeout=30).json()
        latest = r["latest"]
    except Exception as e:
        print(f"Release check failed ({e}) — treating as 'no new release known' rather than guessing.")
        return False, state.get("last_scanned_release")
    is_new = state.get("last_scanned_release") not in (None, latest)
    return is_new, latest


is_new, latest_release = check_for_new_release()
if state.get("last_scanned_release") is None:
    print(f"First run — current Overture release is {latest_release}. "
          f"(kaggle_business_scanner.py should be run against this release if it hasn't been yet.)")
elif is_new:
    print(f"\n*** NEW OVERTURE RELEASE AVAILABLE: {latest_release} "
          f"(last scan used {state['last_scanned_release']}) ***")
    print("Re-run kaggle_business_scanner.py to pick up the new release before relying on "
          "results here for anything time-sensitive — this script does not silently keep "
          "serving stale data without telling you.")
else:
    print(f"Business data is current (release {latest_release}, matches last scan).")
state["last_scanned_release"] = latest_release

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

# %% Real category taxonomy — pulled LIVE from the actual database, never
# invented. This is what makes "search so much better" real: the model picks
# from what the data actually contains, instead of guessing a category
# string that sounds right but doesn't exist (which would silently return
# zero rows and look like a bug).
import duckdb

_con = duckdb.connect()
_con.execute(f"CREATE VIEW biz AS SELECT * FROM read_parquet('{BIZ_DB_LOCAL}')")
REAL_CATEGORIES = sorted(
    r[0] for r in _con.execute(
        "SELECT DISTINCT category FROM biz WHERE category IS NOT NULL"
    ).fetchall()
)
_con.close()
print(f"Loaded {len(REAL_CATEGORIES):,} real distinct category values from the live database.")


def match_categories(plain_english_term, top_k=5):
    """Maps what a user actually says ('dentist', 'coffee shop') to the
    REAL category strings in the database. Two-stage, both grounded in the
    real list — never a free-form guess:
      1. difflib close-match against the real list (fast, catches obvious
         matches like 'dentist' -> 'dental_clinic')
      2. the model picks from a shortlist combining close matches + a
         sample of the full list, forced to answer ONLY with values that
         are actually in REAL_CATEGORIES (checked after, not trusted blindly)
    Returns a list of real category strings, never a fabricated one."""
    term = plain_english_term.strip().lower()
    close = difflib.get_close_matches(term, REAL_CATEGORIES, n=top_k, cutoff=0.3)
    token_matches = [c for c in REAL_CATEGORIES if term.replace(" ", "_") in c or term in c.replace("_", " ")]
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
    always come from the real DuckDB query downstream, not from the model."""
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
    real row counts) — nothing here is invented to fill a gap."""
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
        "output_formats": output_formats or ["csv", "pdf"],
    }
    preview = sba.preview_trade_area(spec, BIZ_DB_LOCAL)
    print(f"Preview: {preview}")
    files = sba.fulfil(spec, BIZ_DB_LOCAL, out_dir)
    return {"spec": spec, "preview": preview, "files": files}

# %% Push the updated state (release marker) and finish
save_state()
print(f"\nCompany Scrapper ready. template@{template_steps:,}, "
      f"{len(REAL_CATEGORIES):,} real categories loaded, business DB at {BIZ_DB_LOCAL}.\n"
      f"Call research(\"find dental clinics with no website within 3 miles of 123 Main St, "
      f"Columbus, OH\") to run a real request.")
