# Model Card Score Extraction via HF MCP Server

This document provides instructions for extracting benchmark scores from HuggingFace model cards using the HF MCP Server tools.

---

## Overview

Model cards often contain evaluation tables with benchmark scores. This guide shows how to:

1. Use `hub_repo_details` to fetch model card content
2. Search for benchmark variations in the README
3. Extract and normalize scores
4. Format results for `.eval_results/`

---

## Step 1: Fetch the Model Card

Use `hub_repo_details` to get the model's README content:

```
mcp__hf-mcp-server__hub_repo_details
  repo_ids: ["org/model-name"]
  include_readme: true
```

This returns:
- Model metadata (downloads, likes, tags, pipeline_tag)
- Full README content (when `include_readme: true`)
- Linked papers and datasets

### Batch Fetching

You can fetch multiple models at once:

```
mcp__hf-mcp-server__hub_repo_details
  repo_ids: ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
  include_readme: true
```

---

## Step 2: Search for Benchmark Scores

### Benchmark Name Variations

Model cards use inconsistent naming. Search for these variations:

| Benchmark | Variations to Search |
|-----------|---------------------|
| HLE | `HLE`, `hle`, `Humanity's Last Exam`, `HLE (Text Only)` |
| GPQA | `GPQA`, `GPQA Diamond`, `gpqa_diamond`, `GPQA-Diamond` |
| MMLU-Pro | `MMLU-Pro`, `MMLU Pro`, `mmlu_pro`, `MMLU-PRO` |
| MMLU | `MMLU`, `mmlu`, `Massive Multitask Language Understanding` |
| GSM8K | `GSM8K`, `gsm8k`, `GSM-8K`, `Grade School Math` |
| HumanEval | `HumanEval`, `humaneval`, `human_eval` |
| HellaSwag | `HellaSwag`, `hellaswag`, `hella_swag` |
| ARC-Challenge | `ARC-Challenge`, `ARC-C`, `arc_challenge` |
| TruthfulQA | `TruthfulQA`, `truthful_qa`, `TruthfulQA MC` |
| MATH | `MATH`, `math`, `MATH-500` |
| AIME | `AIME`, `AIME24`, `AIME 2024`, `aime_24` |
| SWE-bench | `SWE-bench`, `SWE-bench Verified`, `swe_bench` |
| LiveCodeBench | `LiveCodeBench`, `LCB`, `LiveCodeBenchV6` |
| IFEval | `IFEval`, `IF-Eval`, `ifeval` |

---

## Step 3: Identify Table Formats

Model cards typically present scores in these formats:

### Format A: Model-Column Table (most common)
```markdown
| Model | MMLU | GPQA | HLE |
|-------|------|------|-----|
| This Model | 85.2 | 72.1 | 12.3 |
| GPT-4 | 86.4 | 74.2 | 15.1 |
```

### Format B: Benchmark-Column Table
```markdown
| Benchmark | Score |
|-----------|-------|
| MMLU | 85.2 |
| GPQA | 72.1 |
| HLE | 12.3 |
```

### Format C: Inline Text
```markdown
Our model achieves **85.2%** on MMLU, **72.1%** on GPQA Diamond, and **12.3%** on HLE.
```

### Format D: Nested/Grouped Tables
```markdown
| Category | Benchmark | Score |
|----------|-----------|-------|
| Reasoning | GPQA | 72.1 |
| Knowledge | MMLU | 85.2 |
```

---

## Step 4: Extract and Normalize Scores

### Score Format Normalization

Scores may be presented as:
- **Percentages**: `85.2%` or `85.2` (when context implies %)
- **Decimals**: `0.852` (multiply by 100 for percentage)
- **Fractions**: `85.2/100`

**Important**: The `.eval_results/` format expects values matching the benchmark's standard scale. Most benchmarks use percentage scale (0-100).

---

## Step 5: Format for .eval_results/

Once you have the score, format it for `.eval_results/`:

```yaml
# .eval_results/{benchmark}.yaml
- dataset:
    id: cais/hle           # Hub dataset ID (see mapping below)
    task_id: default       # Task variant if applicable
  value: 12.3              # Score value
  date: "2026-01-14"       # ISO date of extraction
  source:
    url: https://huggingface.co/{org}/{model}
    name: Model Card
```

### Dataset ID Reference

| Benchmark | Dataset ID | Task ID |
|-----------|------------|---------|
| HLE | `cais/hle` | `default` |
| GPQA | `Idavidrein/gpqa` | `gpqa_diamond` |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | `default` |
| MMLU | `cais/mmlu` | `default` |
| GSM8K | `openai/gsm8k` | `default` |
| HumanEval | `openai/openai_humaneval` | `default` |
| MATH | `lighteval/MATH` | `default` |
| ARC-Challenge | `allenai/ai2_arc` | `ARC-Challenge` |
| HellaSwag | `Rowan/hellaswag` | `default` |
| TruthfulQA | `truthfulqa/truthful_qa` | `default` |
| SWE-bench | `princeton-nlp/SWE-bench_Verified` | `default` |
| AIME24 | `OpenEvals/aime_24` | `default` |
| AIME25 | `OpenEvals/aime_2025` | `default` |
| LiveCodeBench | `livecodebench/livecodebench` | `default` |
| IFEval | `google/IFEval` | `default` |

---

## Complete Example Workflow

### Scenario: Extract HLE score from a model card

```
1. Fetch model card:
   mcp__hf-mcp-server__hub_repo_details
     repo_ids: ["Qwen/Qwen2.5-72B-Instruct"]
     include_readme: true

2. Search README for HLE variations:
   Found: "| HLE | 18.5 |" in evaluation table

3. Create the eval result:
   $ uv run scripts/evaluation_manager.py add-eval \
       --benchmark HLE \
       --repo-id "Qwen/Qwen2.5-72B-Instruct" \
       --value 18.5 \
       --create-pr
```

---

## Finding Models with Evaluations

Use `model_search` to find models that might have benchmark scores:

### Search for Trending Models
```
mcp__hf-mcp-server__model_search
  task: "text-generation"
  sort: "trendingScore"
  limit: 20
```

### Search by Author
```
mcp__hf-mcp-server__model_search
  author: "meta-llama"
  task: "text-generation"
  limit: 10
```

### Search by Query
```
mcp__hf-mcp-server__model_search
  query: "instruct chat"
  task: "text-generation"
  limit: 20
```

Then use `hub_repo_details` on promising results to check their model cards.

---

## Tips for Better Extraction

### 1. Check the Full README
Model cards may have scores in different sections (Overview, Evaluation, Benchmarks, Results).

### 2. Look for Multiple Tables
Some model cards have separate tables for different benchmark categories.

### 3. Note Evaluation Settings
Papers may report different settings (0-shot vs 5-shot, with/without CoT). Document which setting you're extracting.

### 4. Verify Against Papers
If both paper and model card have scores, prefer the paper as the authoritative source but verify they match.

---

## Common Issues

### Score in Image/Figure Only
**Solution**: Check if there's a linked technical report or paper with tabular data. Use `paper_search` to find it.

### Benchmark Name Differs Significantly
**Solution**: Search for the underlying task name (e.g., "graduate-level science" for GPQA).

### Multiple Scores for Same Benchmark
**Solution**: Prefer "0-shot" or "standard" settings; note the configuration in source attribution.

### Score Not Found
- Score genuinely not present in model card
- Benchmark not evaluated by model authors
- Try `paper_search` to find scores in linked papers

**Solution**: Document why extraction failed to distinguish "not found by automation" from "truly not available".
