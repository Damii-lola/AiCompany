"""
Researchers / Company Scrapper — Chat Understanding worker
==============================================================================
This IS the "addon to the template AI" the user asked for, built the
CORRECT way per standing project rule: every worker is its OWN small LoRA
adapter that stacks on top of the permanent "template" adapter at LOAD
TIME, every run — never a one-time fork that bakes the template's current
state into the worker's weights. Base Qwen2.5-7B-Instruct (4-bit, frozen)
+ "template" adapter (pulled fresh from the repo every run, frozen, never
trained here) + "worker" adapter (THIS worker's own delta — the only
thing this script trains). Both active simultaneously via PEFT's
set_adapter(["template", "worker"]). This is the same shape as every
other worker in this project, just with this worker's own slug/data.

WHY this worker exists at all (read before touching anything else in this
file): the live chat route (render-service/POST /api/chat) used to run on
hand-written regex (render-service/lib/chat.js) for understanding a user's
free-text request — radius/no-website detection, category/address
splitting, follow-up resolution, "near me" detection, delivery-format
detection. That caught real, repeated bugs (typo tolerance, "near me"
having no device location to resolve, noise words like "eligible for
loans" breaking a match, a contentless follow-up getting misread as a
fresh location) because hand-coding a pattern for every phrasing a person
might type is a losing game — there's no way to enumerate "a billion
questions" in regex. This worker's whole job is to replace that
understanding step with a REAL trained model, while the actual data
lookup stays 100% deterministic: this worker is ONLY ever asked to output
a small structured JSON object (what location/category/radius/filters/
delivery-format the user means) — it never sees or reports real business
data, so it can never fabricate a count, a name, or an address. Every
number in every final reply still comes from a live Postgres query run
AFTER this worker's output, by the exact same code (render-service/
server.js's tradeAreaFanOut / nationwide count query) that ran before
this worker existed. This keeps the platform's standing "nothing
fabricated, everything verifiable" rule intact while finally handling
arbitrary phrasing properly.

WORKER_SLUG = "researchers-company-scrapper-chat"
GITHUB_REPO = "Damii-lola/aicompany"  — the repo this worker's template,
    training data, and output adapter all live in (same repo as
    everything else in this project; Kaggle is compute only, never
    storage, per the project's standing Nov 2026 migration decision).

Run this as a Kaggle notebook cell (GPU: T4 x1 or better). Needs a
GITHUB_TOKEN Kaggle Secret (a GitHub PAT with repo write access) to push
the trained adapter back — same mechanism as every other worker/the
template model's own training script.

Output: workers/researchers-company-scrapper-chat/adapter/
    (adapter_config.json, adapter_model.safetensors, README.md) pushed to
    GITHUB_REPO — this is what model_server.py (same directory) loads
    stacked on top of the template to serve live.
"""

import json
import os
import shutil
import subprocess
import sys

WORKER_SLUG = "researchers-company-scrapper-chat"
GITHUB_REPO = "Damii-lola/aicompany"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO}.git"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
WORK_DIR = "/kaggle/working/aicompany"
TEMPLATE_ADAPTER_DIR = f"{WORK_DIR}/model"
WORKER_ADAPTER_DIR = f"{WORK_DIR}/workers/{WORKER_SLUG}/adapter"
TRAINING_DATA_CSV = f"{WORK_DIR}/data/workers/{WORKER_SLUG}/training_examples.csv"

# Same LoRA hyperparameters as the template adapter (model/adapter_config.json)
# so the two adapters are architecturally consistent with each other.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

MAX_SEQ_LEN = 768
NUM_EPOCHS = 6          # this is a small, structured task (one JSON object out)
                        # — a worker adapter like this converges fast on a
                        # few hundred focused examples; more epochs on a
                        # tiny dataset just overfits past this point.
LEARNING_RATE = 2e-4
BATCH_SIZE = 4


def sh(cmd, check=True, **kw):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, **kw)


def clone_or_pull_repo():
    if os.path.isdir(WORK_DIR):
        sh(f"git -C {WORK_DIR} fetch origin main && git -C {WORK_DIR} reset --hard origin/main")
    else:
        sh(f"git clone --depth 1 {GITHUB_REPO_URL} {WORK_DIR}")


def push_adapter():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN Kaggle Secret is not set — cannot push the trained "
            "adapter. Add it under Add-ons -> Secrets in this notebook."
        )
    push_url = GITHUB_REPO_URL.replace("https://", f"https://x-access-token:{token}@")
    sh(f"git -C {WORK_DIR} add workers/{WORKER_SLUG}/adapter")
    # Nothing to commit is not an error (e.g. a re-run that converged to
    # byte-identical weights) — never raise on that specific case.
    diff = subprocess.run(
        f"git -C {WORK_DIR} diff --cached --quiet", shell=True
    ).returncode
    if diff == 0:
        print("No changes to the worker adapter — nothing to push.")
        return
    sh(f'git -C {WORK_DIR} -c user.email="autonomous-ai@local" -c user.name="Autonomous AI" '
       f'commit -m "Train {WORKER_SLUG} worker adapter"')
    sh(f"git -C {WORK_DIR} push {push_url} HEAD:main")


def load_training_examples(path):
    import pandas as pd
    if not os.path.exists(path):
        raise RuntimeError(
            f"No training data found at {path} — this worker needs real "
            f"(context, reply) examples checked into the repo before it can "
            f"train. See data/workers/{WORKER_SLUG}/training_examples.csv."
        )
    df = pd.read_csv(path)
    if "context" not in df.columns or "reply" not in df.columns:
        raise RuntimeError("training_examples.csv must have 'context' and 'reply' columns.")
    df = df.dropna(subset=["context", "reply"])
    if len(df) < 20:
        raise RuntimeError(
            f"Only {len(df)} training examples found — too few for a stable "
            f"adapter. Add more real examples to training_examples.csv."
        )
    print(f"Loaded {len(df)} training examples.")
    return df


SYSTEM_PROMPT_PATH_HINT = (
    # Kept identical, on purpose, to render-service/lib/llm.js's SYSTEM_PROMPT
    # at the time this worker was built — the live server sends the SAME
    # system framing this worker was trained against, so what it learned
    # here matches what it will actually be asked at inference time. If
    # that prompt changes, retrain this worker against the new one.
    "See render-service/lib/llm.js's SYSTEM_PROMPT for the exact instruction "
    "text this worker is trained to follow — kept in sync there, not duplicated here."
)


def build_prompt(tokenizer, context_text):
    # Mirrors the exact chat structure the live server will use at
    # inference time (model_server.py in this same directory) — a system
    # message describing the task/schema, then the user's message (which
    # already contains the short conversation history/context baked in by
    # the caller, same shape render-service sends).
    system_prompt = open(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "SYSTEM_PROMPT.txt")).read() \
        if os.path.exists("SYSTEM_PROMPT.txt") else SYSTEM_PROMPT_PATH_HINT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    sh("pip install -q -U transformers peft bitsandbytes accelerate pandas", check=False)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset

    clone_or_pull_repo()
    df = load_training_examples(TRAINING_DATA_CSV)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")

    # Load the template adapter FROZEN (never trained here — pulled fresh
    # from the repo every run, exactly like every other worker).
    model = PeftModel.from_pretrained(base, TEMPLATE_ADAPTER_DIR, adapter_name="template")
    for p in model.parameters():
        p.requires_grad = False

    # Add (or continue) this worker's OWN adapter.
    if os.path.isdir(WORKER_ADAPTER_DIR) and os.path.exists(f"{WORKER_ADAPTER_DIR}/adapter_config.json"):
        print("Continuing training on this worker's existing adapter.")
        model.load_adapter(WORKER_ADAPTER_DIR, adapter_name="worker", is_trainable=True)
    else:
        print("No existing worker adapter found — creating a fresh one.")
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
        )
        model.add_adapter("worker", lora_config)

    model.set_adapter(["template", "worker"])
    for name, p in model.named_parameters():
        p.requires_grad = "worker" in name and "lora" in name

    def tokenize_row(row):
        prompt = build_prompt(tokenizer, row["context"])
        full = prompt + row["reply"] + tokenizer.eos_token
        enc = tokenizer(full, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"])
        labels = list(enc["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100  # don't train on the prompt tokens, only the JSON reply
        enc["labels"] = labels
        return enc

    ds = Dataset.from_pandas(df[["context", "reply"]].reset_index(drop=True))
    ds = ds.map(tokenize_row, remove_columns=["context", "reply"])

    args = TrainingArguments(
        output_dir=f"{WORK_DIR}/_train_tmp",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()

    os.makedirs(WORKER_ADAPTER_DIR, exist_ok=True)
    model.set_adapter("worker")
    model.save_pretrained(WORK_DIR + "/_adapter_save", selected_adapters=["worker"])
    # save_pretrained with selected_adapters writes to a subfolder named
    # after the adapter — flatten it into the expected WORKER_ADAPTER_DIR.
    saved_sub = WORK_DIR + "/_adapter_save/worker"
    if os.path.isdir(saved_sub):
        for fname in os.listdir(saved_sub):
            shutil.copy(os.path.join(saved_sub, fname), WORKER_ADAPTER_DIR)

    push_adapter()
    print("Done. Worker adapter trained and pushed to", f"workers/{WORKER_SLUG}/adapter/")


if __name__ == "__main__":
    main()
