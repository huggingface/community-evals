# Paper Score Extraction via HF MCP Server

This document provides instructions for extracting benchmark scores from academic papers linked to HuggingFace models using the HF MCP Server tools.

---

## Overview

Papers linked to HuggingFace models often contain comprehensive benchmark results that aren't in the model card. This guide shows how to:

1. Use `hub_repo_details` to discover papers linked to a model
2. Use `paper_search` to find and retrieve paper content
3. Extract benchmark scores from paper abstracts/content
4. Format results for `.eval_results/`

---

## Step 1: Discover Linked Papers

### Get Model Details with README

Use `hub_repo_details` to fetch model metadata including linked papers:

```
mcp__hf-mcp-server__hub_repo_details
  repo_ids: ["org/model-name"]
  include_readme: true
```

Look for arXiv references in:
- `tags` array: entries starting with `arxiv:` (e.g., `"arxiv:2411.15124"`)
- README content: arXiv links or paper references
- Model metadata: `paperInfo` or `cardData.arxiv` fields

### Example Response Fields

The response will include:
- Model metadata (downloads, likes, tags)
- README content (if `include_readme: true`)
- Any linked paper IDs in tags

---

## Step 2: Search for Papers

Once you have an arXiv ID or want to find related papers, use `paper_search`:

```
mcp__hf-mcp-server__paper_search
  query: "OLMo-2 evaluation benchmark"
  results_limit: 5
  concise_only: false  # Get full abstracts for score extraction
```

### Search Strategies

**By model name:**
```
query: "Llama 3.1 benchmark evaluation"
```

**By arXiv ID (if known):**
```
query: "2411.15124"
```

**By benchmark + model family:**
```
query: "MMLU GPQA Qwen2.5"
```

---

## Step 3: Extract Benchmark Scores

The paper search returns abstracts and paper content. Look for:

### Common Benchmark Mentions

Papers typically report headline numbers in abstracts:
- "achieves **85.2%** on MMLU"
- "state-of-the-art results on GPQA Diamond (72.1%)"
- "HLE score of 12.3%"

### Benchmark Name Variations

| Standard Name | Paper Variations |
|---------------|------------------|
| HLE | Humanity's Last Exam, HLE (Text Only) |
| GPQA | GPQA Diamond, GPQA-Diamond |
| MMLU | MMLU, MMLU-Pro, Massive Multitask |
| GSM8K | GSM8K, GSM-8K, Grade School Math |
| MATH | MATH, MATH-500 |
| HumanEval | HumanEval, human_eval |
| SWE-bench | SWE-bench, SWE-bench Verified |

### Score Format Normalization

- **Percentages**: `85.2%` → use `85.2`
- **Decimals**: `0.852` → convert to `85.2` if context shows percentages
- **Accuracy vs Error Rate**: Ensure you're extracting accuracy, not error rate

---

## Step 4: Format for .eval_results/

Once you have extracted scores, format them as YAML:

```yaml
# .eval_results/{benchmark_name}.yaml
- dataset:
    id: {hub_dataset_id}
    task_id: {task_variant}
  value: {score}
  date: "{extraction_date}"
  source:
    url: https://arxiv.org/abs/{arxiv_id}
    name: Paper
```

### Dataset ID Reference

| Benchmark | Dataset ID | Task ID |
|-----------|------------|---------|
| HLE | `cais/hle` | `default` |
| GPQA | `Idavidrein/gpqa` | `gpqa_diamond` |
| MMLU | `cais/mmlu` | `default` |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | `default` |
| GSM8K | `openai/gsm8k` | `default` |
| MATH | `lighteval/MATH` | `default` |
| HumanEval | `openai/openai_humaneval` | `default` |
| DROP | `ucinlp/drop` | `default` |
| ARC-Challenge | `allenai/ai2_arc` | `ARC-Challenge` |
| HellaSwag | `Rowan/hellaswag` | `default` |
| TruthfulQA | `truthfulqa/truthful_qa` | `default` |
| IFEval | `google/IFEval` | `default` |
| SWE-bench | `princeton-nlp/SWE-bench_Verified` | `default` |
| AIME24 | `OpenEvals/aime_24` | `default` |
| AIME25 | `OpenEvals/aime_2025` | `default` |
| LiveCodeBench | `livecodebench/livecodebench` | `default` |

---

## Complete Example Workflow

### Scenario: Extract MMLU score for OLMo-2 from its paper

```
1. Get model details:
   mcp__hf-mcp-server__hub_repo_details
     repo_ids: ["allenai/OLMo-2-1124-7B-Instruct"]
     include_readme: true

   → Found tags: ["arxiv:2411.15124", "arxiv:2501.00656"]

2. Search for the paper:
   mcp__hf-mcp-server__paper_search
     query: "OLMo-2 2501.00656"
     results_limit: 3

   → Returns paper abstract with benchmark scores

3. Extract from abstract:
   "OLMo-2-7B-Instruct achieves 61.3 on MMLU..."

4. Create the eval result:
   $ uv run scripts/evaluation_manager.py add-eval \
       --benchmark MMLU \
       --repo-id "allenai/OLMo-2-1124-7B-Instruct" \
       --value 61.3 \
       --create-pr
```

---

## Tips for Better Extraction

### 1. Check Multiple Papers
Models may have multiple linked papers. Use `hub_repo_details` to find all arXiv tags, then search for each.

### 2. Use Concise Mode for Broad Searches
```
mcp__hf-mcp-server__paper_search
  query: "large language model evaluation"
  concise_only: true  # 2-sentence summaries
  results_limit: 10
```

### 3. Prefer Primary Sources
Use the model's own release paper rather than papers that cite it.

### 4. Note Evaluation Settings
Papers may report different settings (0-shot vs 5-shot, with/without CoT). Document which setting you're extracting.

### 5. Cross-Reference Model Card
If both paper and model card have scores, prefer the paper as the authoritative source but verify they match.

---

## Common Issues

### Paper Score Differs from Model Card
- Paper may report different model size/variant
- Evaluation settings may differ
- Paper may be pre-release; model card updated post-release

**Solution**: Note the discrepancy and prefer the most recent source.

### Score Not Found in Paper Search
- Paper may not have evaluated that benchmark
- Try searching with different query terms
- Check if benchmark uses a different name

**Solution**: Try alternative query terms, search for benchmark aliases.

### Multiple Models in Paper
- Paper describes a family of models (7B, 13B, 70B)
- Results may combine scores across sizes

**Solution**: Carefully match the exact model variant to the HuggingFace repo.
