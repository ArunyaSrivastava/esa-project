#!/usr/bin/env bash
# Publish esa-project as a fork under the user's GitHub account.
# Usage:  ./publish_fork.sh [commit message]
# Requires: gh CLI authenticated (gh auth login) OR git push credentials.

set -euo pipefail

GITHUB_USER="ArunyaSrivastava"
UPSTREAM="amorfati3735/esa-project"
FORK_REPO="https://github.com/${GITHUB_USER}/esa-project.git"
COMMIT_MSG="${1:-Capstone updates: BLE wearable, modular dashboard, docs}"

cd "$(dirname "$0")"

echo "==> Ensuring fork exists under ${GITHUB_USER}..."
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh repo fork "${UPSTREAM}" --clone=false || echo "    (fork may already exist)"
else
    echo "    gh not authenticated — assuming the fork already exists on GitHub."
    echo "    Create it at: https://github.com/${UPSTREAM}/fork"
fi

echo "==> Configuring git remotes..."
if git remote get-url fork >/dev/null 2>&1; then
    git remote set-url fork "${FORK_REPO}"
else
    git remote add fork "${FORK_REPO}"
fi
git remote set-url origin "https://github.com/${UPSTREAM}.git"

echo "==> Committing changes..."
git add -A
if git diff --cached --quiet; then
    echo "    Nothing to commit."
else
    git commit -m "${COMMIT_MSG}"
fi

echo "==> Pushing to your fork (branch: $(git branch --show-current))..."
git push -u fork "$(git branch --show-current)"

echo
echo "✅ Done. Your fork: https://github.com/${GITHUB_USER}/esa-project"
