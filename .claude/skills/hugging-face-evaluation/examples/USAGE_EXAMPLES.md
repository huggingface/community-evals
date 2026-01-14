# Usage Examples

Practical examples for adding evaluations to HuggingFace model repositories using the `.eval_results/` format.

## Table of Contents
1. [Setup](#setup)
2. [Add Single Benchmark (Recommended)](#add-single-benchmark-recommended)
3. [Batch Process Trending Models](#batch-process-trending-models)
4. [Extract from README Tables](#extract-from-readme-tables)
5. [Import from Artificial Analysis](#import-from-artificial-analysis)
6. [Common Workflows](#common-workflows)

---

## Setup

### Environment Variables

```bash
# Required for creating PRs
export HF_TOKEN="hf_your_write_token_here"

# Optional: for Artificial Analysis source
export AA_API_KEY="your_aa_api_key_here"
```

Or use a `.env` file:
```bash
cp examples/.env.example .env
# Edit .env with your tokens
```

### Verify Installation

```bash
uv run scripts/evaluation_manager.py --help
```

---

## Add Single Benchmark (Recommended)

The simplest way to add a specific benchmark score to a model.

### Basic Usage

```bash
# Preview (default - prints YAML without uploading)
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "moonshotai/Kimi-K2-Thinking"
```

Output:
```
Looking up HLE score for moonshotai/Kimi-K2-Thinking from model_card...
Found: HLE = 23.9
Generated YAML:
- dataset:
    id: cais/hle
    task_id: default
  value: 23.9
  date: "2026-01-14"
  source:
    url: https://huggingface.co/moonshotai/Kimi-K2-Thinking
    name: Model Card
```

### From Artificial Analysis

```bash
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "MiniMaxAI/MiniMax-M2.1" \
  --source aa
```

### Create PR

```bash
# Always check for existing PRs first!
uv run scripts/evaluation_manager.py get-prs --repo-id "model/name"

# If no PRs exist, create one
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "model/name" \
  --create-pr
```

### Push Directly (Your Own Model)

```bash
uv run scripts/evaluation_manager.py add-eval \
  --benchmark GPQA \
  --repo-id "your-username/your-model" \
  --apply
```

### Provide Score Manually

```bash
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "model/name" \
  --value 84.5 \
  --create-pr
```

---

## Batch Process Trending Models

Process multiple trending models at once.

### Preview Mode (Dry Run)

```bash
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE --dry-run
```

Output:
```
==================================================
Batch Evaluation PR Creator
==================================================
Benchmark: HLE
Source: model_card
Pipeline tag: text-generation
Limit: 10
Sort: trending
Dry run: True
==================================================

Processing: LiquidAI/LFM2.5-1.2B-Instruct
  Not found: HLE score not available
Processing: MiniMaxAI/MiniMax-M2.1
  Found: HLE = 22.2
  Status: Would create PR (dry run)
...

Summary:
Success: 3
Not found: 7
```

### Create PRs

```bash
# From model cards
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE

# From Artificial Analysis
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE --source aa
```

### Sort Options

```bash
# By downloads (more established models)
uv run scripts/batch_eval_prs.py --limit 20 --sort downloads --benchmark GPQA

# By likes
uv run scripts/batch_eval_prs.py --limit 10 --sort likes --benchmark MMLU-Pro
```

### Filter by Pipeline Tag

```bash
# Only text-generation models (default)
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE --pipeline-tag text-generation

# Image generation models
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE --pipeline-tag text-to-image
```

### Results Tracking

Results are saved to `runs/{benchmark}_{date}_{hash}.json`:

```bash
cat runs/hle_20260114_abc123.json
```

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

---

## Extract from README Tables

For models with evaluation tables in their README.

### Step 1: Inspect Tables

```bash
uv run scripts/evaluation_manager.py inspect-tables \
  --repo-id "deepseek-ai/DeepSeek-V3"
```

This shows all tables with their structure, helping you identify which table to extract.

### Step 2: Preview Extraction

```bash
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "deepseek-ai/DeepSeek-V3" \
  --table 1
```

### Step 3: Create PR

```bash
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "deepseek-ai/DeepSeek-V3" \
  --table 1 \
  --create-pr
```

---

## Import from Artificial Analysis

Import all available benchmarks from Artificial Analysis API.

### Preview

```bash
uv run scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "your-username/claude-mirror"
```

### Create PR

```bash
uv run scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "your-username/claude-mirror" \
  --apply --create-pr
```

### Finding Creator Slug and Model Name

Visit [Artificial Analysis](https://artificialanalysis.ai/) and check the URL:
- URL: `https://artificialanalysis.ai/models/{creator-slug}/{model-name}`

Common examples:
- Anthropic: `--creator-slug "anthropic" --model-name "claude-sonnet-4"`
- OpenAI: `--creator-slug "openai" --model-name "gpt-4-turbo"`
- Meta: `--creator-slug "meta" --model-name "llama-3-70b"`

---

## Common Workflows

### Workflow 1: Add Missing Benchmark to Popular Model

```bash
# 1. Check for existing PRs
uv run scripts/evaluation_manager.py get-prs \
  --repo-id "meta-llama/Llama-3.1-8B-Instruct"

# 2. Preview what we'd add
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "meta-llama/Llama-3.1-8B-Instruct"

# 3. Create PR if score found
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "meta-llama/Llama-3.1-8B-Instruct" \
  --create-pr
```

### Workflow 2: Batch Update Trending Models

```bash
# 1. Dry run to see which models have HLE scores
uv run scripts/batch_eval_prs.py --limit 20 --benchmark HLE --source aa --dry-run

# 2. Create PRs for models with scores
uv run scripts/batch_eval_prs.py --limit 20 --benchmark HLE --source aa

# 3. Check results
cat runs/hle_*.json | jq '.results[] | select(.status == "pr_created")'
```

### Workflow 3: Update Your Own Model

```bash
# 1. Add HLE score from your model card
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "your-username/your-model" \
  --apply

# 2. Add GPQA score manually
uv run scripts/evaluation_manager.py add-eval \
  --benchmark GPQA \
  --repo-id "your-username/your-model" \
  --value 84.5 \
  --apply
```

---

## Output Format

Results are stored in `.eval_results/*.yaml`:

```yaml
- dataset:
    id: cais/hle              # Hub Benchmark dataset ID
    task_id: default          # Optional task ID
  value: 23.9                 # Metric value
  date: "2026-01-14"          # ISO-8601 date
  source:                     # Attribution
    url: https://huggingface.co/model/name
    name: Model Card
```

---

## Supported Benchmarks

| Benchmark | Hub Dataset ID |
|-----------|---------------|
| HLE | cais/hle |
| GPQA | Idavidrein/gpqa |
| MMLU-Pro | TIGER-Lab/MMLU-Pro |
| GSM8K | openai/gsm8k |

To add a new benchmark, update `examples/metric_mapping.json`.

---

## Troubleshooting

### "AA_API_KEY not set"
```bash
export AA_API_KEY="your-key"
# or add to .env file
```

### "Could not find benchmark in model card"
The benchmark name may be formatted differently in the README. Check the model card manually.

### "Token does not have write access"
Generate a new token at https://huggingface.co/settings/tokens with Write scope.

---

## Getting Help

```bash
uv run scripts/evaluation_manager.py --help
uv run scripts/evaluation_manager.py add-eval --help
uv run scripts/batch_eval_prs.py --help
```

For more information:
- [HuggingFace Eval Results Documentation](https://huggingface.co/docs/hub/eval-results)
- [SKILL.md](../SKILL.md) - Complete skill documentation
