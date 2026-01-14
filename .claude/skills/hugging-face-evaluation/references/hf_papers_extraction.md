# Manual Paper Score Extraction

This document provides instructions for manually extracting benchmark scores from academic papers linked to HuggingFace models when automated tools are unavailable or fail.

**Use this approach when**:
- The `evaluation_manager.py extract-paper` command is not available
- You need more control over the extraction process
- The automated tool failed to find specific benchmarks

---

## Overview

Papers linked to HuggingFace models often contain comprehensive benchmark results that aren't in the model card. This guide shows how to:

1. Discover papers linked to a model
2. Download and extract text from paper PDFs
3. Use Claude to extract benchmark scores
4. Format results for `.eval_results/`

---

## Step 1: Discover Linked Papers

### Option A: Check HuggingFace API

```bash
# Fetch model metadata
curl -s "https://huggingface.co/api/models/{org}/{model}" | python -m json.tool
```

Look for arXiv references in:
- `tags` array: entries starting with `arxiv:` (e.g., `"arxiv:2411.15124"`)
- `cardData.arxiv`: direct arxiv ID field
- `paperInfo`: array of paper metadata

### Option B: Check the Model Card

Visit `https://huggingface.co/{org}/{model}` and look for:
- "Paper" or "Technical Report" links in the header
- arXiv links in the README content
- References section at the bottom

### Option C: Use WebFetch

```
WebFetch: https://huggingface.co/api/models/{org}/{model}
Prompt: Find all arxiv paper IDs linked to this model. Look in tags (format: arxiv:XXXX.XXXXX), cardData.arxiv field, and paperInfo array.
```

---

## Step 2: Download Paper PDF

Once you have an arXiv ID (e.g., `2411.15124`), construct the PDF URL:

```
https://arxiv.org/pdf/{arxiv_id}.pdf
```

Example:
```bash
# Download PDF
curl -L "https://arxiv.org/pdf/2411.15124.pdf" -o paper.pdf
```

### Using WebFetch for Paper Content

For quick extraction without downloading:

```
WebFetch: https://arxiv.org/abs/2411.15124
Prompt: Extract the abstract and find any links to evaluation results or benchmark scores mentioned.
```

Note: WebFetch works better with the abstract page (`/abs/`) than the PDF.

---

## Step 3: Extract Benchmark Scores

### Method A: Direct Claude Prompting (Recommended)

If you have the paper text, prompt Claude directly:

```
I have a paper about the model "{model_name}". Please extract all benchmark evaluation scores for this model.

Look for common benchmarks like:
- MMLU, MMLU-Pro
- GPQA, GPQA Diamond
- GSM8K, MATH
- HumanEval, MBPP
- HLE (Humanity's Last Exam)
- ARC-Challenge, HellaSwag
- TruthfulQA, IFEval
- DROP, SQuAD

Return the scores as a structured list with:
- Benchmark name (as written in the paper)
- Score value (numeric, without % symbol)
- Any relevant notes (e.g., "0-shot", "5-shot")

Paper content:
{paper_text}
```

### Method B: Using WebFetch on arXiv HTML

```
WebFetch: https://arxiv.org/abs/2411.15124
Prompt: Extract all benchmark scores and evaluation results for the main model described in this paper. List each benchmark name and its corresponding score.
```

### Method C: Section-by-Section Extraction

For long papers, focus on specific sections:

1. **Abstract**: Often mentions headline benchmark numbers
2. **Results/Experiments section**: Contains detailed tables
3. **Appendix**: May have additional benchmark breakdowns

Prompt example:
```
Focus on the "Experiments" or "Results" section of this paper.
Extract all benchmark scores reported for {model_name}.
Present as: Benchmark Name: Score
```

---

## Step 4: Interpret Common Table Formats

### Format A: Comparison Table (Most Common)

Papers often compare against baselines:

```
| Model        | MMLU | GPQA | GSM8K |
|--------------|------|------|-------|
| GPT-4        | 86.4 | 53.6 | 92.0  |
| Our Model    | 85.1 | 51.2 | 89.5  |  <- Extract this row
| Llama-3      | 79.2 | 46.1 | 84.2  |
```

**Key**: Identify which row corresponds to the model being evaluated.

### Format B: Per-Task Breakdown

```
| Benchmark    | Setting | Score |
|--------------|---------|-------|
| MMLU         | 5-shot  | 85.1  |
| MMLU         | 0-shot  | 82.3  |
| GSM8K        | CoT     | 89.5  |
```

**Key**: Note the evaluation setting (shots, chain-of-thought, etc.)

### Format C: Aggregated Categories

```
| Category     | Benchmarks           | Avg Score |
|--------------|----------------------|-----------|
| Reasoning    | GPQA, ARC-C, BBH     | 72.4      |
| Math         | GSM8K, MATH          | 85.2      |
```

**Key**: Look for individual scores elsewhere or use category averages.

---

## Step 5: Format for .eval_results/

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

---

## Complete Example Workflow

### Scenario: Extract MMLU score for OLMo-2 from its paper

```
1. Find linked papers:
   $ curl -s "https://huggingface.co/api/models/allenai/OLMo-2-1124-7B-Instruct" | grep -o '"arxiv:[^"]*"'
   > "arxiv:2411.15124"
   > "arxiv:2501.00656"

2. Check paper for benchmark tables:
   WebFetch: https://arxiv.org/abs/2501.00656
   Prompt: Extract MMLU score for OLMo-2-7B-Instruct from this paper

3. Response shows: MMLU = 61.3

4. Create the eval result:
   $ uv run scripts/evaluation_manager.py add-eval \
       --benchmark MMLU \
       --repo-id "allenai/OLMo-2-1124-7B-Instruct" \
       --value 61.3 \
       --create-pr
```

### Alternative: Pure Manual Approach

If you can't use the script, create the YAML file directly:

```yaml
# .eval_results/mmlu.yaml
- dataset:
    id: cais/mmlu
  value: 61.3
  date: "2026-01-14"
  source:
    url: https://arxiv.org/abs/2501.00656
    name: Paper
```

Then submit via HuggingFace PR:
1. Fork the model repository
2. Add the file to `.eval_results/mmlu.yaml`
3. Create a pull request

---

## Tips for Better Extraction

### 1. Check Multiple Papers
Models may have multiple papers (technical report, follow-up studies). Check all linked papers.

### 2. Prefer Primary Sources
Use the model's own release paper rather than papers that cite it.

### 3. Note Evaluation Settings
Papers may report different settings (0-shot vs 5-shot, with/without CoT). Document which setting you're extracting.

### 4. Cross-Reference Model Card
If both paper and model card have scores, prefer the paper as the authoritative source but verify they match.

### 5. Handle Score Formats
- Percentages: `85.2%` -> use `85.2`
- Decimals: `0.852` -> convert to `85.2` if other scores in paper are percentages
- Accuracy vs Error Rate: Ensure you're extracting accuracy, not error rate

---

## Common Issues

### Paper Score Differs from Model Card
- Paper may report different model size/variant
- Evaluation settings may differ
- Paper may be pre-release; model card updated post-release

**Solution**: Note the discrepancy and prefer the most recent source.

### Score Not Found for Specific Benchmark
- Paper may not have evaluated that benchmark
- Benchmark may be in appendix or supplementary materials
- Benchmark may use a different name

**Solution**: Check appendices, search for benchmark aliases.

### Multiple Models in Paper
- Paper describes a family of models (7B, 13B, 70B)
- Tables may combine results across sizes

**Solution**: Carefully match the exact model variant to the HuggingFace repo.
