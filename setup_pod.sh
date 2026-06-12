#!/usr/bin/env bash
# One-time pod setup: install Claude Code, create the venv, and make every
# new terminal activate it automatically. Safe to re-run (idempotent).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 0. Make sure ~/.local/bin is on the PATH (claude + uv install there) ----
export PATH="$HOME/.local/bin:$PATH"
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if ! grep -qF "$PATH_LINE" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "# Add ~/.local/bin to PATH (added by setup_pod.sh)"
        echo "$PATH_LINE"
    } >> "$HOME/.bashrc"
    echo ">> Added ~/.local/bin to PATH in ~/.bashrc"
fi

# --- 1. Claude Code ---------------------------------------------------------
if ! command -v claude >/dev/null 2>&1; then
    echo ">> Installing Claude Code..."
    curl -fsSL https://claude.ai/install.sh | bash
else
    echo ">> Claude Code already installed: $(claude --version)"
fi

# --- 2. uv + venv -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo ">> Creating venv and installing dependencies (uv sync)..."
cd "$REPO_DIR"
uv sync

# --- 3. Git identity ----------------------------------------------------------
echo ">> Setting git user..."
git config --global user.email "michaellaret7@gmail.com"
git config --global user.name "michael laret"

# --- 4. Local tracking branches for all remote branches ----------------------
echo ">> Setting up local branches tracking origin..."
git fetch --all --prune
for remote in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin | grep -v '^origin/HEAD$'); do
    branch="${remote#origin/}"
    if ! git show-ref --verify --quiet "refs/heads/$branch"; then
        git branch --track "$branch" "$remote"
        echo ">> Created local branch $branch tracking $remote"
    fi
done

# --- 5. Auto-activate venv in every new terminal ----------------------------
ACTIVATE_LINE="source \"$REPO_DIR/.venv/bin/activate\""
if ! grep -qF "$ACTIVATE_LINE" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "# Auto-activate RLVHR venv (added by setup_pod.sh)"
        echo "$ACTIVATE_LINE"
    } >> "$HOME/.bashrc"
    echo ">> Added venv activation to ~/.bashrc"
else
    echo ">> venv activation already in ~/.bashrc"
fi

echo ">> Done. Open a new terminal (or run: source ~/.bashrc) to activate the venv."
