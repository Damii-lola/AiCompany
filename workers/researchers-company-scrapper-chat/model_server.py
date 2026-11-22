"""
Live inference server for the researchers-company-scrapper-chat worker.

Loads: base Qwen2.5-7B-Instruct (4-bit, frozen) + "template" adapter
(pulled fresh from the repo every boot, frozen) + "worker" adapter
(trained by train_worker.py in this same directory) stacked together via
PEFT's set_adapter(["template", "worker"]). Exposes ONE endpoint:

    POST /understand
    body: {"message": "...", "context": {...} | null}
    returns: the structured JSON object described in SYSTEM_PROMPT.txt
             (same file train_worker.py trains against), plus a top-level
             "ok": true/false and, on failure, "error".

This process NEVER touches real business data and never talks to
Postgres — it only turns free text into a structured intent object.
render-service (Node) is the thing that takes this JSON and runs the
real, deterministic database query, same as it always has.

----------------------------------------------------------------------
WHERE THIS ACTUALLY RUNS (the real infrastructure decision)
----------------------------------------------------------------------
This script needs a GPU and needs to stay warm enough to answer a chat
message in a few seconds — a Kaggle notebook is not that (it's not a
persistent server you can POST to), and GitHub Actions has no GPU
(already ruled out for this exact reason earlier in this project).

Concrete answer: deploy this on Modal (modal.com) using modal_deploy.py
in this same directory. Modal is the right fit here specifically because
it is serverless (no server to babysit, scales to zero between messages
so it costs ~nothing when idle — matching the rest of this project's
free/low-cost-by-default posture), it runs plain Python/PyTorch (no
separate packaging step), and its free tier includes real GPU minutes
that comfortably cover a chat worker's usage pattern (bursty, a few
seconds per message, not constantly running). `modal deploy modal_deploy.py`
gives a stable HTTPS URL to put in render-service's MODEL_SERVER_URL env
var — that's the entire integration.

This file (model_server.py) is still useful standalone too: it's a plain
FastAPI app, so it also runs as-is on any other GPU host (RunPod, Lambda,
a rented GPU box, etc.) with `uvicorn model_server:app`, if a different
host is ever preferred later. modal_deploy.py just wraps the same
load/infer functions below for Modal's specific deployment shape.
----------------------------------------------------------------------
"""

import json
import os
import re

from fastapi import FastAPI
from pydantic import BaseModel

WORKER_SLUG = "researchers-company-scrapper-chat"
GITHUB_REPO = "Damii-lola/aicompany"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO}.git"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
WORK_DIR = os.environ.get("AICOMPANY_REPO_DIR", "/tmp/aicompany")
TEMPLATE_ADAPTER_DIR = f"{WORK_DIR}/model"
WORKER_ADAPTER_DIR = f"{WORK_DIR}/workers/{WORKER_SLUG}/adapter"
SYSTEM_PROMPT_PATH = f"{WORK_DIR}/workers/{WORKER_SLUG}/SYSTEM_PROMPT.txt"

app = FastAPI()

_model = None
_tokenizer = None
_system_prompt = None


class UnderstandRequest(BaseModel):
    message: str
    context: dict | None = None


def _ensure_repo_fresh():
    # The template adapter and this worker's own adapter both live in the
    # GitHub repo and must be pulled FRESH on every boot (never baked into
    # a Docker layer / snapshot) — that's the whole point of stacking
    # adapters at load time instead of forking: if the template model
    # gets retrained/improved, this worker picks that up automatically on
    # its next cold start with zero code change here.
    # An authenticated clone URL (e.g. for a private repo) can be supplied
    # via AICOMPANY_REPO_URL_OVERRIDE (modal_deploy.py sets this from the
    # github-token Modal Secret) without hardcoding a token in this file.
    clone_url = os.environ.get("AICOMPANY_REPO_URL_OVERRIDE", GITHUB_REPO_URL)
    if os.path.isdir(WORK_DIR):
        os.system(f"git -C {WORK_DIR} fetch origin main && git -C {WORK_DIR} reset --hard origin/main")
    else:
        os.system(f"git clone --depth 1 {clone_url} {WORK_DIR}")


def load_model():
    global _model, _tokenizer, _system_prompt
    if _model is not None:
        return

    _ensure_repo_fresh()

    if not os.path.exists(f"{WORKER_ADAPTER_DIR}/adapter_config.json"):
        raise RuntimeError(
            f"No trained worker adapter found at {WORKER_ADAPTER_DIR} — run "
            f"train_worker.py on Kaggle first and push its output, then "
            f"redeploy this server."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base, TEMPLATE_ADAPTER_DIR, adapter_name="template")
    model.load_adapter(WORKER_ADAPTER_DIR, adapter_name="worker")
    model.set_adapter(["template", "worker"])
    model.eval()

    _model = model
    with open(SYSTEM_PROMPT_PATH) as f:
        _system_prompt = f.read()


def _build_user_text(message, context):
    if context:
        return "Prior context: " + json.dumps(context, ensure_ascii=False) + "\nNew message: " + message
    return message


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

REQUIRED_FIELDS = {
    "intent", "category_phrase", "address", "radius_miles",
    "no_website_only", "delivery_mode", "unsupported_note", "clarifying_reply",
}


def run_inference(message: str, context: dict | None):
    import torch

    load_model()
    user_text = _build_user_text(message, context)
    chat = [
        {"role": "system", "content": _system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = _tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        out = _model.generate(
            **inputs, max_new_tokens=256, do_sample=False, temperature=None, top_p=None,
            pad_token_id=_tokenizer.pad_token_id,
        )
    text = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    match = _JSON_OBJ_RE.search(text)
    if not match:
        raise ValueError(f"Model did not return a JSON object. Raw output: {text!r}")
    obj = json.loads(match.group(0))
    missing = REQUIRED_FIELDS - set(obj.keys())
    if missing:
        raise ValueError(f"Model output missing required fields: {missing}")
    if obj["intent"] not in ("search", "count", "clarify", "unsupported"):
        raise ValueError(f"Model output has an invalid intent: {obj['intent']!r}")
    return obj


@app.post("/understand")
def understand(req: UnderstandRequest):
    try:
        result = run_inference(req.message, req.context)
        return {"ok": True, **result}
    except Exception as e:  # noqa: BLE001 — this endpoint must never 500 silently;
        # render-service's caller treats any non-ok response as "fall back
        # to the deterministic parser", so surfacing the real error here
        # (rather than crashing the process) keeps that fallback working.
        return {"ok": False, "error": str(e)}


@app.get("/healthz")
def healthz():
    return {"ok": True, "model_loaded": _model is not None}
