#!/usr/bin/env bash
# setup_scheduled_jobs.sh
#
# Schedule both monitoring scripts as nightly Hugging Face Jobs.
# Run this once to register the schedules; the jobs will then execute
# automatically at 2 AM UTC every day.
#
# Prerequisites:
#   - huggingface-cli installed and logged in (`huggingface-cli login`)
#   - HF_TOKEN secret available in your HF organisation/user secrets
#
# Usage:
#   bash scripts/setup_scheduled_jobs.sh

set -euo pipefail

DATASET_ID="burtenshaw/community-evals-monitoring"

# Raw script URLs from the GitHub repo main branch
POLL_SCRIPT_URL="https://raw.githubusercontent.com/burtenshaw/community-evals/main/scripts/poll_new_evals.py"
AUDIT_SCRIPT_URL="https://raw.githubusercontent.com/burtenshaw/community-evals/main/scripts/audit_model_coverage.py"

# ---------------------------------------------------------------------------
# 1. Poll new evals — runs nightly at 2 AM UTC
#    Reads/writes poll_state.json on the Hub so state survives between runs.
# ---------------------------------------------------------------------------
hf jobs scheduled run "0 2 * * *" \
    --flavor cpu-basic \
    --secrets HF_TOKEN \
    -- \
    uv run "${POLL_SCRIPT_URL}" \
        --dataset-id "${DATASET_ID}"

echo "Scheduled: poll_new_evals.py (daily at 02:00 UTC)"

# ---------------------------------------------------------------------------
# 2. Audit model coverage — runs nightly at 2 AM UTC
#    Pushes audit_results/YYYY-MM-DD.json and audit_results/latest.json.
# ---------------------------------------------------------------------------
hf jobs scheduled run "0 2 * * *" \
    --flavor cpu-basic \
    --secrets HF_TOKEN \
    -- \
    uv run "${AUDIT_SCRIPT_URL}" \
        --dataset-id "${DATASET_ID}" \
        --output json

echo "Scheduled: audit_model_coverage.py (daily at 02:00 UTC)"
