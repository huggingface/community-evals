---
name: hugging-face-evaluation
description: Add evaluation results to Hugging Face model repositories using the .eval_results/ format. Uses HF CLI for PR management and manual YAML creation.
---

# Overview

This skill adds structured evaluation results to HuggingFace model repositories using the [`.eval_results/` format](https://huggingface.co/docs/hub/eval-results).

**What This Enables:**
- Results appear on model pages with benchmark links
- Scores are aggregated into benchmark dataset leaderboards
- Community contributions via Pull Requests

# Important

Evaluation PRs can only be opened on the Hugging Face Hub. They cannot be opened on the GitHub repository.

# Version
3.0.0

# Workflow Overview

The actual workflow uses:
1. **HF CLI** (`hf upload`, `hf download`) for PR operations
2. **Manual YAML creation** in `/tmp/pr-reviews/`
3. **`check_prs.py`** script to check for existing PRs
4. **curl** to fetch model cards and leaderboard data

See `references/hf_cli_for_prs.md` for detailed CLI instructions.

---

# CRITICAL: Multiple Scores for One Benchmark

Models can have multiple scores for the same benchmark (with/without tools). **Each variant MUST be in a separate file.**

## File Naming Convention

| Condition | File Name | Notes Field |
|-----------|-----------|-------------|
| Default (no tools) | `hle.yaml` | None (omit notes) |
| With tools | `hle_with_tools.yaml` | `notes: "With tools"` |

## Notes Field Rules

1. **No tools = No notes field** - Default assumption is "without tools"
2. **With tools = Add notes** - Only add when tools ARE used
3. **Standardized format** - Always use `notes: "With tools"` (capital W)

**CORRECT:**
```yaml
# hle.yaml (no tools - DEFAULT)
- dataset:
    id: cais/hle
    task_id: hle
  value: 22.1
  source:
    url: https://huggingface.co/org/model
    name: Model Card
    user: username
```

```yaml
# hle_with_tools.yaml (with tools)
- dataset:
    id: cais/hle
    task_id: hle
  value: 44.9
  source:
    url: https://huggingface.co/org/model
    name: Model Card
    user: username
  notes: "With tools"
```

**INCORRECT:**
```yaml
notes: "Without tools"  # Don't add notes for default
notes: "w/ tools"       # Use standardized format
notes: "with tools"     # Capital W required
```

---

# Core Workflow

## Step 1: Check for Existing PRs

**ALWAYS check before creating new PRs:**

```bash
uv run scripts/check_prs.py --repo-id "org/model-name"
```

If PRs exist, update them instead of creating new ones.

## Step 2: Fetch Model Card and Extract Scores

```bash
# Get model README
curl -s "https://huggingface.co/org/model-name/raw/main/README.md" | grep -i -A10 "HLE\|GPQA\|MMLU"
```

Or use MCP tools:
```
mcp__hf-mcp-server__hub_repo_details
  repo_ids: ["org/model-name"]
  include_readme: true
```

## Step 3: Create YAML File

```bash
mkdir -p /tmp/pr-reviews/new-prs
cd /tmp/pr-reviews/new-prs

cat > hle.yaml << 'EOF'
- dataset:
    id: cais/hle
    task_id: hle
  value: 22.1
  date: '2026-02-03'
  source:
    url: https://huggingface.co/org/model-name
    name: Model Card
    user: burtenshaw
EOF
```

## Step 4: Create PR

```bash
hf upload org/model-name hle.yaml .eval_results/hle.yaml \
  --repo-type model --create-pr \
  --commit-message "Add HLE evaluation result"
```

## Step 5: Get PR Number

```bash
uv run scripts/check_prs.py --repo-id "org/model-name"
```

---

# Updating Existing PRs

```bash
# Download PR contents
hf download org/model-name --repo-type model \
  --revision refs/pr/<PR_NUMBER> \
  --include ".eval_results/*" \
  --local-dir /tmp/pr-reviews/model-pr<PR_NUMBER>

# Edit the YAML file, then upload
hf upload org/model-name /tmp/pr-reviews/updated.yaml .eval_results/hle.yaml \
  --repo-type model \
  --revision refs/pr/<PR_NUMBER> \
  --commit-message "Update evaluation result"
```

---

# Deleting Files from PRs

Use Python API:
```bash
uv run --with huggingface_hub python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.delete_file(
    path_in_repo=".eval_results/old_file.yaml",
    repo_id="org/model-name",
    repo_type="model",
    revision="refs/pr/<PR_NUMBER>",
    commit_message="Remove file"
)
EOF
```

---

# Fetching Leaderboard Data

```bash
# HLE leaderboard (requires auth for private datasets)
curl -s "https://huggingface.co/api/datasets/cais/hle/leaderboard" \
  -H "Authorization: Bearer $HF_TOKEN"

# MMLU-Pro leaderboard (public)
curl -s "https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro/leaderboard"

# Model eval results
curl -s "https://huggingface.co/api/models/org/model?expand[]=evalResults"
```

---

# .eval_results/ Format

```yaml
# .eval_results/hle.yaml
- dataset:
    id: cais/hle              # Required: Hub Benchmark dataset ID
    task_id: hle              # Required: task id from dataset's eval.yaml
  value: 22.2                 # Required: metric value
  date: "2026-01-14"          # Optional: ISO-8601 date
  source:                     # Optional: attribution
    url: https://huggingface.co/org/model
    name: Model Card
    user: username
```

---

# Supported Benchmarks

| Benchmark | Hub Dataset ID | Task ID |
|-----------|---------------|---------|
| HLE | cais/hle | hle |
| GPQA | Idavidrein/gpqa | diamond |
| MMLU-Pro | TIGER-Lab/MMLU-Pro | mmlu_pro |

---

# Tool-Using Agent Models

Models like MiroThinker, Nemotron-Orchestrator are inherently tool-using agents. For these:

1. Use `hle_with_tools.yaml` as filename
2. Add `notes: "With tools"`
3. Look for terms: "search agent", "agentic", "orchestrator", "code-interpreter"

---

# Environment Setup

```bash
export HF_TOKEN="your-huggingface-token"
```

---

# Scripts Reference

```bash
# Check for existing PRs (ALWAYS do this first)
uv run scripts/check_prs.py --repo-id "org/model-name"
```

See `references/hf_cli_for_prs.md` for complete HF CLI workflow documentation.

---

# Best Practices

1. **Always check for existing PRs** before creating new ones
2. **Separate files for variants** - `hle.yaml` for default, `hle_with_tools.yaml` for tools
3. **Notes only for non-default** - Omit notes for standard evaluations
4. **Standardized format** - Use `"With tools"` exactly (capital W)
5. **Verify scores** - Compare YAML against model card before submitting
