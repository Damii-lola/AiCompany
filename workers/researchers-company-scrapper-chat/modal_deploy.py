"""
Deploys model_server.py's /understand endpoint on Modal (modal.com) — a
serverless GPU host that scales to zero between requests, so this costs
nothing while idle and only spins up a GPU for the few seconds it takes
to answer one chat message. This is the concrete, named answer to "where
does the fine-tuned worker actually run live", since neither Kaggle
notebooks (not a persistent server) nor GitHub Actions (no GPU, already
ruled out) can serve this.

One-time setup (run these once, from any machine with Python):
    pip install modal
    modal setup                     # opens a browser to link your Modal account (free tier is enough to start)
    modal secret create github-token GITHUB_TOKEN=<a GitHub PAT, read access is enough here>

Deploy / redeploy (e.g. after retraining the worker adapter on Kaggle and
pushing it to the repo):
    modal deploy modal_deploy.py

That prints a stable HTTPS URL like:
    https://<your-modal-username>--company-scrapper-chat-understand.modal.run
Put that URL in render-service's MODEL_SERVER_URL environment variable
(Render dashboard -> Environment) and redeploy render-service — that's
the whole integration; render-service already falls back to its
deterministic parser automatically if this URL is unset, unreachable, or
slow, so nothing else breaks while this is being set up.
"""

import modal

app = modal.App("company-scrapper-chat")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "transformers", "peft", "bitsandbytes", "accelerate",
        "fastapi[standard]",
    )
)

WORK_DIR = "/tmp/aicompany"


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("github-token")],
    scaledown_window=120,  # stay warm 2 minutes after the last message so a
                           # back-and-forth chat doesn't cold-start every turn
    timeout=120,
)
@modal.asgi_app()
def understand_app():
    import os
    os.environ["AICOMPANY_REPO_DIR"] = WORK_DIR

    # model_server.py's _ensure_repo_fresh() does a plain `git clone`/
    # `fetch` — give it an authenticated remote URL so it also works for
    # a private repo, by rewriting the URL it uses via an env override.
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        os.environ["AICOMPANY_REPO_URL_OVERRIDE"] = (
            f"https://x-access-token:{token}@github.com/Damii-lola/aicompany.git"
        )

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_server import app as fastapi_app  # noqa: E402

    return fastapi_app
