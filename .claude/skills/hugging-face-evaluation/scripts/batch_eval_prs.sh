#!/usr/bin/env bash
# /// script
# requires: huggingface-hub>=1.3.1
# ///
#
# Batch create evaluation PRs for trending HuggingFace models.
#
# Usage:
#   ./batch_eval_prs.sh [OPTIONS]
#
# Options:
#   --limit N          Number of models to process (default: 10)
#   --sort FIELD       Sort by: downloads, likes, trending (default: trending)
#   --benchmark NAME   Benchmark to add (default: HLE)
#   --source SOURCE    Score source: model_card, aa (default: model_card)
#   --dry-run          Preview without creating PRs
#   --results FILE     JSON file to track results (default: eval_results.json)
#
# Examples:
#   ./batch_eval_prs.sh --limit 5 --benchmark HLE --dry-run
#   ./batch_eval_prs.sh --limit 20 --sort trending --benchmark GPQA
#   ./batch_eval_prs.sh --limit 10 --benchmark MMLU --source aa

set -euo pipefail

# Default values
LIMIT=10
SORT="trending"
BENCHMARK="HLE"
SOURCE="model_card"
DRY_RUN=false
RESULTS_FILE="eval_results.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --sort)
            SORT="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --results)
            RESULTS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for required tools
if ! command -v hf &> /dev/null; then
    echo "Error: huggingface-hub CLI not found. Install with: pip install 'huggingface-hub>=1.3.1'"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq not found. Install with: brew install jq (macOS) or apt install jq (Linux)"
    exit 1
fi

# Initialize results file if it doesn't exist
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo '{
  "created": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",
  "benchmark": "'$BENCHMARK'",
  "source": "'$SOURCE'",
  "results": []
}' > "$RESULTS_FILE"
fi

echo "========================================"
echo "Batch Evaluation PR Creator"
echo "========================================"
echo "Benchmark: $BENCHMARK"
echo "Source: $SOURCE"
echo "Limit: $LIMIT"
echo "Sort: $SORT"
echo "Dry run: $DRY_RUN"
echo "Results file: $RESULTS_FILE"
echo "========================================"
echo

# Get trending models
echo "Fetching top $LIMIT models (sorted by $SORT)..."
MODELS=$(hf models ls --sort "$SORT" --limit "$LIMIT" --json 2>/dev/null | jq -r '.[].id')

if [[ -z "$MODELS" ]]; then
    echo "No models found or error fetching models"
    exit 1
fi

echo "Found models:"
echo "$MODELS" | head -5
if [[ $(echo "$MODELS" | wc -l) -gt 5 ]]; then
    echo "... and $(($(echo "$MODELS" | wc -l) - 5)) more"
fi
echo

# Process each model
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

while IFS= read -r REPO_ID; do
    [[ -z "$REPO_ID" ]] && continue

    echo "----------------------------------------"
    echo "Processing: $REPO_ID"

    # Check if already processed
    if jq -e ".results[] | select(.repo_id == \"$REPO_ID\" and .benchmark == \"$BENCHMARK\")" "$RESULTS_FILE" > /dev/null 2>&1; then
        echo "  Skipping: Already processed"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # Try to add eval
    if [[ "$DRY_RUN" == "true" ]]; then
        OUTPUT=$(uv run "$SCRIPT_DIR/evaluation_manager.py" add-eval \
            --benchmark "$BENCHMARK" \
            --repo-id "$REPO_ID" \
            --source "$SOURCE" 2>&1) || true
    else
        OUTPUT=$(uv run "$SCRIPT_DIR/evaluation_manager.py" add-eval \
            --benchmark "$BENCHMARK" \
            --repo-id "$REPO_ID" \
            --source "$SOURCE" \
            --create-pr 2>&1) || true
    fi

    # Parse result
    if echo "$OUTPUT" | grep -q "Found:"; then
        VALUE=$(echo "$OUTPUT" | grep "Found:" | sed 's/.*= //')
        echo "  Found: $BENCHMARK = $VALUE"

        if [[ "$DRY_RUN" == "true" ]]; then
            STATUS="dry_run"
            echo "  Status: Would create PR (dry run)"
        elif echo "$OUTPUT" | grep -q "Pull request created\|uploaded successfully"; then
            STATUS="pr_created"
            echo "  Status: PR created"
        else
            STATUS="uploaded"
            echo "  Status: Uploaded"
        fi

        # Add to results
        TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        jq ".results += [{
            \"repo_id\": \"$REPO_ID\",
            \"benchmark\": \"$BENCHMARK\",
            \"value\": $VALUE,
            \"source\": \"$SOURCE\",
            \"status\": \"$STATUS\",
            \"timestamp\": \"$TIMESTAMP\"
        }]" "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"

        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  Not found: $BENCHMARK score not available"

        # Record as not_found
        TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        jq ".results += [{
            \"repo_id\": \"$REPO_ID\",
            \"benchmark\": \"$BENCHMARK\",
            \"value\": null,
            \"source\": \"$SOURCE\",
            \"status\": \"not_found\",
            \"timestamp\": \"$TIMESTAMP\"
        }]" "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"

        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

done <<< "$MODELS"

echo
echo "========================================"
echo "Summary"
echo "========================================"
echo "Processed: $((SUCCESS_COUNT + FAIL_COUNT + SKIP_COUNT))"
echo "Success: $SUCCESS_COUNT"
echo "Not found: $FAIL_COUNT"
echo "Skipped: $SKIP_COUNT"
echo "Results saved to: $RESULTS_FILE"
echo "========================================"
