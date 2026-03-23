# community-evals

A repository for community-driven evaluation results for open source models on HuggingFace Hub.

This project adds structured evaluation results to model repositories using the new [`.eval_results/` format](https://huggingface.co/docs/hub/eval-results), enabling:

- Benchmark Leaderboards: Results appear on both model pages and benchmark dataset leaderboards
- Community Contributions: Anyone can submit evaluation results via Pull Request
- Verified Results: Peer reviewed and verified evaluation results

![Model Evaluation Results](https://huggingface.co/huggingface/documentation-images/resolve/main/evaluation-results/eval-results-previw.png)

## Supported Benchmarks

| Benchmark | Hub Dataset ID |
|-----------|---------------|
| HLE | [cais/hle](https://huggingface.co/datasets/cais/hle) |
| GPQA | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) |
| MMLU-Pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |

## Quick Start

### Prerequisites

```bash
# Set environment variables (or add to .env file)
export HF_TOKEN="your-huggingface-token"  # Required: write access for PRs
```

### Add a Single Evaluation

```bash
# Preview (dry run)
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "moonshotai/Kimi-K2-Thinking"

# Create PR
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "moonshotai/Kimi-K2-Thinking" \
  --create-pr
```

### Batch Process Trending Models

```bash
# Preview top 10 trending LLMs
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE --dry-run

# Create PRs for models with scores
uv run scripts/batch_eval_prs.py --limit 10 --benchmark HLE
```

---

## Manual Usage (specific model and benchmark)

### Step 1: Look Up Score (Preview)

```bash
# From model card (default)
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "model/name"

# From Artificial Analysis
uv run scripts/evaluation_manager.py add-eval \
  --benchmark GPQA \
  --repo-id "model/name" \
  --source aa
```

**Sources:**
- `model_card` (default): Extract from README tables
- `aa`: Query Artificial Analysis API (requires `AA_API_KEY`)
- `papers`: HuggingFace Papers (not yet implemented)

### Step 2: Check Existing PRs

Always check for existing PRs before creating new ones:

```bash
uv run scripts/evaluation_manager.py get-prs --repo-id "model/name"
```

### Step 3: Apply Changes

```bash
# Create PR (someone else's model)
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "other-user/their-model" \
  --create-pr

# Push directly (model authors)
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "your-username/your-model" \
  --apply
```

## Manual Usage (get all evaluations for a model card)

For models with evaluation tables in their README:

```bash
# 1. Inspect available tables
uv run scripts/evaluation_manager.py inspect-tables --repo-id "model/name"
```

This will output the available tables and their numbers which you can then use to extract the evaluations.

---

## Agentic Usage

This repository includes a Claude Code skill at `.claude/skills/hugging-face-evaluation/` for automated evaluation management.

### Ideal Prompts

**Add specific benchmark to a model:**

```
Add the HLE evaluation results to moonshotai/Kimi-K2-Thinking from the model card
```

**Batch process trending models:**

```
Find the top 20 trending text-generation models and add HLE scores from Model Cards. Do a dry run first.
```

```
Find the top 20 trending text-generation models and add HLE scores from Artificial Analysis. Do a dry run first.
```

**Check and add multiple benchmarks:**

```
For meta-llama/Llama-3.1-8B-Instruct, check the model card for GPQA and MMLU-Pro scores and create PRs if found.
```

**Extract from README tables:**

```
Inspect the evaluation tables in deepseek-ai/DeepSeek-V3 and extract all benchmark scores to .eval_results/ format
```

### Workflow for Agents

1. **Check for existing PRs** before creating new ones
2. **Preview first** (dry run) to verify scores are found
3. **Use appropriate source** (model_card for README tables, aa for Artificial Analysis)
4. **Create PRs** for models you don't own, push directly for your own

### Example Agent Session

```
User: Add HLE score to MiniMaxAI/MiniMax-M2.1 from Artificial Analysis

Agent:
1. Checks for existing open PRs on MiniMaxAI/MiniMax-M2.1
2. Looks up HLE score from Artificial Analysis API
3. Finds: HLE = 22.2%
4. Creates PR with .eval_results/hle.yaml containing the score
5. Returns PR URL to user
```

---

## Output Format

Results are stored as YAML files in `.eval_results/`:

```yaml
# .eval_results/hle.yaml
- dataset:
    id: cais/hle              # Hub Benchmark dataset ID
    task_id: default          # Optional: specific task/leaderboard
  value: 22.2                 # Metric value
  date: "2026-01-14"          # ISO-8601 date
  source:                     # Attribution
    url: https://artificialanalysis.ai
    name: Artificial Analysis
```

See [HuggingFace Eval Results Documentation](https://huggingface.co/docs/hub/eval-results) for full format specification.

---

## Run Results

Results from batch runs are saved to `runs/` directory:

```
runs/
├── hle_20260114_abc123.json
├── gpqa_20260114_def456.json
└── ...
```

Each file contains:
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

**Status values:** `pr_created`, `uploaded`, `not_found`, `dry_run`, `error`
