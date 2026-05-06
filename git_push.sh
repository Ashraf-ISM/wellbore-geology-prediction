#!/usr/bin/env bash

# =========================================================
# Professional Git Auto Push Script
# Author : Md Ashraf
# Purpose: Auto add, commit, pull-rebase, and push to GitHub
# =========================================================

set -euo pipefail

# ---------- Colors ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---------- Functions ----------
print_header() {
    echo -e "${CYAN}"
    echo "================================================="
    echo "         GitHub Auto Push Utility"
    echo "================================================="
    echo -e "${NC}"
}

error_exit() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

success_msg() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

info_msg() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

warn_msg() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# ---------- Start ----------
print_header

# Check if git exists
command -v git >/dev/null 2>&1 || error_exit "Git is not installed."

# Check if inside git repo
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    error_exit "Current directory is not a Git repository."

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)

info_msg "Current Branch: ${CURRENT_BRANCH}"

# Show git status
echo
git status --short
echo

# Check if there are changes
if [[ -z $(git status --porcelain) ]]; then
    warn_msg "No changes detected to commit."
    exit 0
fi

# Ask commit message
read -rp "Enter commit message: " COMMIT_MSG

# Validate commit message
if [[ -z "$COMMIT_MSG" ]]; then
    error_exit "Commit message cannot be empty."
fi

# Timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Final commit message
FINAL_MSG="${COMMIT_MSG} | ${TIMESTAMP}"

info_msg "Adding files..."
git add .

success_msg "Files staged."

# Commit
info_msg "Creating commit..."
git commit -m "$FINAL_MSG" || warn_msg "Nothing new to commit."

# Pull latest changes safely
info_msg "Pulling latest remote changes..."
git pull origin "$CURRENT_BRANCH" --rebase || \
    error_exit "Git pull failed. Resolve conflicts manually."

# Push
info_msg "Pushing to GitHub..."
git push origin "$CURRENT_BRANCH"

success_msg "Code pushed successfully!"
echo

# Show latest commit
git log --oneline -1

echo
success_msg "Workflow completed."