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

# HF MCP Server Tools

This skill uses the HF MCP Server for model and paper discovery. Key tools:

- **`hub_repo_details`**: Fetch model metadata and README content
  ```
  mcp__hf-mcp-server__hub_repo_details
    repo_ids: ["org/model-name"]
    include_readme: true
  ```

- **`paper_search`**: Search ML papers on HuggingFace
  ```
  mcp__hf-mcp-server__paper_search
    query: "model name benchmark"
    results_limit: 5
  ```

- **`model_search`**: Find models by task, author, or trending
  ```
  mcp__hf-mcp-server__model_search
    task: "text-generation"
    sort: "trendingScore"
    limit: 20
  ```

See `references/hf_papers_extraction.md` and `references/model_card_extraction.md` for detailed usage.

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

# Update Existing PRs (Preferred)

When a PR already exists (especially one authored by the user), **update that PR instead of opening a new one**. PRs on the Hub live in `refs/pr/<NUMBER>` and must be pushed back to that ref. See: [Hub PRs & refs](https://huggingface.co/docs/hub/en/repositories-pull-requests-discussions).

**Workflow (search → edit → push):**

1. **Search for open PRs and filter by author**
   ```bash
   uv run scripts/evaluation_manager.py get-prs --repo-id "org/model-name"
   ```
   - From the output, pick PRs where `Author` matches the user.
   - If multiple PRs exist, update the one that already touches `.eval_results/`.

2. **Download the PR ref locally using the HF CLI**
   ```bash
   hf download org/model-name --repo-type model \
     --revision refs/pr/<PR_NUMBER> \
     --local-dir /tmp/model-pr-<PR_NUMBER>
   ```

3. **Get and edit the eval YAML**
   - File(s) live in `/tmp/model-pr-<PR_NUMBER>/.eval_results/*.yaml`.
   - If adding a new field or entry (e.g., extra metadata in `eval.yaml`), update the existing YAML rather than creating a second file unless asked.

4. **Upload changes back to the PR ref**
   ```bash
   hf upload org/model-name /tmp/model-pr-<PR_NUMBER>/.eval_results/eval.yaml .eval_results/eval.yaml \
     --repo-type model \
     --revision refs/pr/<PR_NUMBER> \
     --commit-message "Update eval results metadata"
   ```
   This updates the existing PR on the Hub. See: [Managing PRs locally](https://huggingface.co/docs/hub/en/repositories-pull-requests-discussions).

**Programmatic option:** use `huggingface_hub` to update files and create or update PRs from Python when needed.
See: [Create or edit PRs programmatically](https://huggingface.co/docs/huggingface_hub/v1.3.3/en/guides/community#create-and-edit-a-discussion-or-pull-request-programmatically).

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
- `model_card` (default): Extract from README tables (use `hub_repo_details` MCP tool)
- `aa`: Query Artificial Analysis API (requires `AA_API_KEY`)
- Manual: Extract from linked papers using `paper_search` MCP tool

## 2. List Open Eval PRs

Find all open PRs on HuggingFace that contain evaluation results (.eval_results/).

```bash
# Scan trending models for open eval PRs
uv run scripts/list_eval_prs.py --limit 30 --verbose

# Filter by PR author
uv run scripts/list_eval_prs.py --user nielsr --pretty

# Filter by model pattern
uv run scripts/list_eval_prs.py --model "meta-llama/*"

# Include merged PRs
uv run scripts/list_eval_prs.py --limit 50 --include-merged
```

**Output JSON format:**
```json
[
  {
    "user": "nielsr",
    "date": "2026-01-15T19:26:35+00:00",
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "pr_num": 332,
    "pr_title": "Add community evaluation results for GPQA, MMLU-PRO, GSM8K",
    "pr_status": "open",
    "dataset_id": "gpqa",
    "eval_yaml_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/refs%2Fpr%2F332/.eval_results/gpqa.yaml"
  }
]
```

## 3. Batch Process Trending Models

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

**Dry Run Output Table Format:**

When reporting dry run results to users, present findings as a markdown table with these columns:

| Column | Description |
|--------|-------------|
| Link | HuggingFace model URL as markdown link |
| Score | Benchmark score (percentage or `-` if not found) |
| Source | Where score was found: `Model Card`, `Paper`, `Artificial Analysis`, or `Not found` |
| Comments | Additional context (e.g., "with tools", "agentic", "reasoning mode") |

Example output:
```markdown
| Link | Score | Source | Comments |
|------|-------|--------|----------|
| [org/model-name](https://hf.co/org/model-name) | **24.8%** | Model Card | 42.8% with tools |
| [org/other-model](https://hf.co/org/other-model) | - | Not found | Code-focused model |
```

**Note:** The automated script may miss scores in non-standard table formats. For comprehensive results, also use `hub_repo_details` with `include_readme: true` to manually inspect model cards for benchmark tables.

## 4. Get Top AA Models

Fetch the top 10 models from the Artificial Analysis index. Use the existing instructions in this skill to match Hub repos, check non-AA sources first, and propose PRs in this priority: **model card → papers → Artificial Analysis**.

```bash
AA_API_KEY=... uv run scripts/aa_top_models_prs.py --limit 10 --pretty
```

**Output fields:**
- `aa_name`, `aa_slug`, `aa_creator`, `aa_index_score`

## 5. Extract from README Tables

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

## 6. Extract from Papers

For models with linked papers on HuggingFace, use the HF MCP Server tools:

### Step 1: Find linked papers
```
mcp__hf-mcp-server__hub_repo_details
  repo_ids: ["meta-llama/Llama-3.1-8B-Instruct"]
  include_readme: true
```
Look for `arxiv:` tags in the response.

### Step 2: Search for paper content
```
mcp__hf-mcp-server__paper_search
  query: "Llama 3.1 2407.21783"
  results_limit: 3
```

### Step 3: Extract scores and create PR
```bash
uv run scripts/evaluation_manager.py add-eval \
  --benchmark MMLU \
  --repo-id "meta-llama/Llama-3.1-8B-Instruct" \
  --value 73.5 \
  --create-pr
```

**How it works:**
1. Use `hub_repo_details` to find arxiv paper IDs in model tags
2. Use `paper_search` to retrieve paper abstracts with benchmark scores
3. Extract scores from paper content and create eval results

See `references/hf_papers_extraction.md` for detailed instructions.

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
    task_id: default          # Required: task id from the dataset's eval.yaml
    revision:                 # Optional: dataset revision (commit hash)
  value: 22.2                 # Required: metric value
  verifyToken:                # Optional: cryptographic proof (Inspect + HF Jobs)
  date: "2026-01-14"          # Optional: ISO-8601 date
  source:                     # Optional: attribution
    url: https://artificialanalysis.ai
    name: Artificial Analysis
    user: my-org              # Optional: HF username/org (ASK if unknown)
```

**Minimal example:**
```yaml
- dataset:
    id: Idavidrein/gpqa
    task_id: gpqa_diamond
  value: 0.412
```

**Agent instruction:** always include `source.user` when the contributing user/org is known. If it is not known, ask for it before submitting the PR.

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

**Find available benchmark datasets on the Hub:**
```bash
hf datasets ls --filter "benchmark:eval-yaml"
```

---

# Find Models With Eval Results

Use the Hub CLI to list models that already publish `.eval_results/`:

```bash
hf models ls --filter "eval-results"
```

---

# Get Eval Results From Hub APIs

Use Hub REST endpoints to fetch aggregated benchmark leaderboards and per-model eval results.

**Benchmark leaderboard (dataset repo):**
```
https://huggingface.co/api/datasets/cais/hle/leaderboard
```
Pattern:
```
/api/:repoType(datasets)/:namespace/:repo/leaderboard
```

**Model eval results (model repo):**
```
https://huggingface.co/api/models/zai-org/GLM-4.7?expand[]=evalResults
```

Pattern:
```
/api/:repoType(models)/:namespace/:repo?expand[]=evalResults
```

---

# Commands Reference

```bash
# List all open PRs with eval results across HuggingFace
uv run scripts/list_eval_prs.py --limit 30 --verbose
uv run scripts/list_eval_prs.py --user nielsr --pretty
uv run scripts/list_eval_prs.py --model "meta-llama/*" --pretty
uv run scripts/list_eval_prs.py --limit 50 --include-merged

# Get top AA models (use model_card → papers → AA priority after)
AA_API_KEY=... uv run scripts/aa_top_models_prs.py --limit 10 --pretty

# Check for existing PRs (ALWAYS do this first)
uv run scripts/evaluation_manager.py get-prs --repo-id "model/name"

# Add single benchmark
uv run scripts/evaluation_manager.py add-eval \
  --benchmark HLE \
  --repo-id "model/name" \
  [--source model_card|aa] \
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

# Extract from linked papers (use HF MCP Server tools first)
# See references/hf_papers_extraction.md for MCP-based workflow

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
→ The model doesn't have any arxiv papers linked in its metadata; use `hub_repo_details` to check tags

**"Paper search returns no results"**
→ Try different query terms (model name, arxiv ID, benchmark name)

---

# Best Practices

1. **Always check for existing PRs** before creating new ones
2. **Preview first** - default behavior prints YAML without uploading
3. **Use dry-run** for batch processing to verify which models have scores
4. **Create PRs** for models you don't own; use `--apply` for your own
5. **Verify scores** - compare output against source before submitting
6. **Track results** - use the `--runs-dir` option to track results and never delete run logs.
