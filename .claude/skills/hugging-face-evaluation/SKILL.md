---
name: hugging-face-evaluation
description: Add and manage evaluation results in Hugging Face model repositories using the new .eval_results/ format. Supports extracting scores from model cards, importing from Artificial Analysis API, and batch processing trending models.
---

# Overview

This skill adds structured evaluation results to HuggingFace model repositories using the [`.eval_results/` format](https://huggingface.co/docs/hub/eval-results).

**What This Enables:**
- Results appear on model pages with benchmark links
- Scores are aggregated into benchmark dataset leaderboards
- Community contributions via Pull Requests
- Verification of evaluation runs

![Model Evaluation Results](https://huggingface.co/huggingface/documentation-images/resolve/main/evaluation-results/eval-results-previw.png)

# Version
2.2.0

# Dependencies

Core dependencies are auto-installed via PEP 723 headers when using `uv run`:
- huggingface_hub>=0.20.0
- markdown-it-py>=3.0.0
- python-dotenv>=1.0.0
- pyyaml>=6.0
- requests>=2.31.0
- pypdf>=4.0.0 (for paper extraction)

# IMPORTANT: Check for Existing PRs

**Before creating ANY pull request, ALWAYS check for existing open PRs:**

```bash
uv run scripts/evaluation_manager.py get-prs --repo-id "username/model-name"
```

**If open PRs exist:**
1. **DO NOT create a new PR** - this creates duplicate work for maintainers
2. **Warn the user** about existing PRs
3. **Show the PR URLs** so they can review them
4. Only proceed if user explicitly confirms

---

# Core Workflows

## 1. Add Single Benchmark (Recommended)

Add a specific benchmark score to a model from various sources.

```bash
# Preview (default - prints YAML)
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "moonshotai/Kimi-K2-Thinking"

# From Artificial Analysis
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "model/name" \
  --source aa

# Create PR
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "model/name" \
  --create-pr

# Push directly (your own model)
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "your-username/your-model" \
  --apply
```

**Sources:**
- `model_card` (default): Extract from README tables
- `aa`: Query Artificial Analysis API (requires `AA_API_KEY`)
- `papers`: Extract from linked HuggingFace Papers (uses Claude Code CLI)

## 2. Batch Process Trending Models

Process multiple models at once using the HuggingFace API.

```bash
# Preview (dry run)
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE --dry-run

# Create PRs for trending LLMs
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE

# Use Artificial Analysis as source
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE --source aa

# Sort by downloads instead of trending
uv run scripts/batch_eval_prs.py --limit 20 --sort downloads --benchmark GPQA

# Filter by pipeline tag (default: text-generation)
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE --pipeline-tag text-generation
```

**Options:**
- `--limit N`: Number of models to process
- `--benchmark NAME`: Benchmark to add (HLE, GPQA, MMLU-Pro, etc.)
- `--source SOURCE`: Score source (model_card, aa)
- `--sort FIELD`: Sort by trending (default), downloads, or likes
- `--pipeline-tag TAG`: Filter by pipeline tag (default: text-generation)
- `--dry-run`: Preview without creating PRs
- `--runs-dir DIR`: Directory for results (default: repo root/runs/)

**Results Tracking:**

Results are saved to `runs/{benchmark}_{date}_{hash}.json`:
```json
{
  "benchmark": "HLE",
  "source": "aa",
  "source_url": "https://artificialanalysis.ai",
  "created": "2026-01-14T08:00:00Z",
  "results": [
    {
      "repo_id": "MiniMaxAI/MiniMax-M2.1",
      "value": 22.2,
      "status": "pr_created",
      "source_url": "https://artificialanalysis.ai"
    }
  ]
}
```

Status values: `pr_created`, `uploaded`, `not_found`, `dry_run`, `error`

## 3. Extract from README Tables

For models with evaluation tables in their README:

```bash
# 1. Inspect tables to find structure
uv run scripts/evaluation_manager.py inspect-tables --repo-id "model/name"

# 2. Extract specific table (prints YAML)
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "model/name" \
  --table 1

# 3. Create PR
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "model/name" \
  --table 1 \
  --create-pr
```

## 4. Extract from Papers

For models with linked papers on HuggingFace (arxiv papers linked in model card):

```bash
# Preview scores from linked papers
uv run scripts/evaluation_manager.py extract-paper \
  --repo-id "meta-llama/Llama-3.1-8B-Instruct"

# Create PR with extracted scores
uv run scripts/evaluation_manager.py extract-paper \
  --repo-id "model/name" \
  --create-pr

# Or use as source for single benchmark
uv run scripts/evaluation_manager.py add-eval \
  --benchmark MMLU \
  --repo-id "model/name" \
  --source papers
```

**How it works:**
1. Fetches papers linked to the model via HuggingFace API
2. Downloads and extracts text from paper PDFs (arXiv)
3. Uses Claude Code CLI to intelligently extract benchmark scores
4. Converts scores to `.eval_results/` format

**Requirements:**
- Claude Code CLI must be installed and available in PATH
- HF_TOKEN for accessing model info (optional but recommended)

---

# Environment Setup

```bash
# Required for creating PRs
export HF_TOKEN="your-huggingface-token"

# Optional: for Artificial Analysis source
export AA_API_KEY="your-aa-api-key"

# Or use .env file
echo "HF_TOKEN=your-token" >> .env
echo "AA_API_KEY=your-aa-key" >> .env
```

---

# .eval_results/ Format

Results are stored as YAML files in `.eval_results/`:

```yaml
# .eval_results/hle.yaml
- dataset:
    id: cais/hle              # Required: Hub Benchmark dataset ID
    task_id: default          # Optional: specific task/leaderboard
  value: 22.2                 # Required: metric value
  date: "2026-01-14"          # Optional: ISO-8601 date
  source:                     # Optional: attribution
    url: https://artificialanalysis.ai
    name: Artificial Analysis
```

**Minimal example:**
```yaml
- dataset:
    id: Idavidrein/gpqa
    task_id: gpqa_diamond
  value: 0.412
```

**Result Badges:**
| Badge | Condition |
|-------|-----------|
| verified | Valid `verifyToken` (ran in HF Jobs with inspect-ai) |
| community-provided | Result submitted via open PR |
| leaderboard | Links to benchmark dataset |
| source | Links to evaluation logs |

---

# Supported Benchmarks

Benchmarks are mapped via `examples/metric_mapping.json`:

| Benchmark | Hub Dataset ID | Task ID |
|-----------|---------------|---------|
| HLE | cais/hle | default |
| GPQA | Idavidrein/gpqa | gpqa_diamond |
| MMLU-Pro | TIGER-Lab/MMLU-Pro | - |
| GSM8K | openai/gsm8k | - |

To add a new benchmark, update `examples/metric_mapping.json`.

---

# Commands Reference

```bash
# Check for existing PRs (ALWAYS do this first)
uv run scripts/evaluation_manager.py get-prs --repo-id "model/name"

# Add single benchmark
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "model/name" \
  [--source model_card|aa|papers] \
  [--value 84.5] \
  [--apply | --create-pr]

# Batch process trending models
uv run scripts/batch_eval_prs.py \
  --limit N \
  --benchmark NAME \
  [--source model_card|aa] \
  [--sort trending|downloads|likes] \
  [--pipeline-tag text-generation] \
  [--dry-run]

# Inspect README tables
uv run scripts/evaluation_manager.py inspect-tables --repo-id "model/name"

# Extract from README table
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "model/name" \
  --table N \
  [--apply | --create-pr]

# Extract from linked papers (uses Claude Code)
uv run scripts/evaluation_manager.py extract-paper \
  --repo-id "model/name" \
  [--apply | --create-pr]

# View current evaluations
uv run scripts/evaluation_manager.py show --repo-id "model/name"

# Validate format
uv run scripts/evaluation_manager.py validate --repo-id "model/name"

# Get help
uv run scripts/evaluation_manager.py --help
uv run scripts/evaluation_manager.py add-eval --help
```

---

# Troubleshooting

**"AA_API_KEY not set"**
→ Set environment variable or add to .env file

**"Token does not have write access"**
→ Ensure HF_TOKEN has write permissions

**"No evaluation tables found in README"**
→ Check if README contains markdown tables with numeric scores

**"Could not find benchmark in model card"**
→ The benchmark name may be formatted differently; check the README manually

**"Model not found in Artificial Analysis"**
→ Not all models are tracked by AA; try `--source model_card` instead

**"No papers found linked to model"**
→ The model doesn't have any arxiv papers linked in its metadata

**"Claude CLI not found"**
→ Install Claude Code CLI and ensure `claude` is in your PATH

**"Error extracting text from PDF"**
→ The paper PDF may be image-based or have complex formatting; try another source

---

# Advanced: Run Custom Evaluations

For running evaluations (not just importing existing scores), see the vLLM and inspect-ai scripts in `scripts/`:

- `inspect_eval_uv.py` - Run inspect-ai evaluations via HF Jobs
- `lighteval_vllm_uv.py` - Run lighteval with vLLM backend
- `inspect_vllm_uv.py` - Run inspect-ai with vLLM backend
- `run_eval_job.py` - Helper for submitting HF Jobs
- `run_vllm_eval_job.py` - Helper for vLLM job submission

These require GPU hardware and are for generating new evaluation results, not importing existing ones.

---

# Best Practices

1. **Always check for existing PRs** before creating new ones
2. **Preview first** - default behavior prints YAML without uploading
3. **Use dry-run** for batch processing to verify which models have scores
4. **Create PRs** for models you don't own; use `--apply` for your own
5. **Verify scores** - compare output against source before submitting
6. **Track results** - use the `--runs-dir` option to track results and never delete run logs.
