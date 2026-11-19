#!/usr/bin/env bash
# One-time setup for a fresh Oracle Always Free Ampere A1 VM (Ubuntu).
# Run this ONCE, right after your first SSH login:
#   bash bootstrap_oracle_vm.sh
#
# Installs Docker + the Compose plugin, clones the repo, and leaves you
# ready to create your real .env file and run docker compose up. Safe to
# re-run (every step checks before acting) if you ever need to repair a VM.

set -euo pipefail

REPO_URL="https://github.com/Damii-lola/AiCompany.git"
REPO_DIR="$HOME/AiCompany"

echo "== Updating packages =="
sudo apt-get update -y
sudo apt-get upgrade -y

echo "== Installing Docker Engine + Compose plugin =="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo "Added $USER to the docker group — log out and back in (or run 'newgrp docker') "
  echo "for this to take effect without needing sudo on every docker command."
else
  echo "Docker already installed, skipping."
fi

echo "== Cloning the repo =="
if [ -d "$REPO_DIR" ]; then
  echo "$REPO_DIR already exists — pulling latest instead of re-cloning."
  git -C "$REPO_DIR" pull
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo
echo "===================================================================="
echo "Bootstrap complete. Next steps:"
echo "  1. cd $REPO_DIR/deploy"
echo "  2. cp company-scrapper-scanner.env.example company-scrapper-scanner.env"
echo "     then edit that real .env with your actual Prisma Postgres values"
echo "     (nano company-scrapper-scanner.env)"
echo "  3. docker compose up -d --build"
echo "  4. docker compose logs -f company-scrapper-scanner   # watch it work"
echo "===================================================================="
