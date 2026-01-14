# Manual Model Card Score Extraction

This document provides instructions for manually extracting benchmark scores from HuggingFace model cards when the automated `evaluation_manager.py` script fails to find scores.

**Use this as a last resort** after the automated tools return "score not available".

---

## When to Use Manual Extraction

Use manual extraction when:
1. `evaluation_manager.py add-eval --source model_card` returns "not available"
2. You suspect the model card contains the score but in an unusual format
3. The benchmark name varies from standard naming conventions

---

## Step 1: Fetch the Model Card README

Use the HuggingFace Hub CLI or API to fetch the raw README content:

```bash
# Using huggingface-cli (recommended)
huggingface-cli download {org}/{model} README.md --local-dir /tmp/model-card
cat /tmp/model-card/README.md

# Using curl (direct raw URL)
curl -s "https://huggingface.co/{org}/{model}/raw/main/README.md"

# Using huggingface_hub Python library
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('{org}/{model}', 'README.md')
print(open(path).read())
"
```

Or use WebFetch to read the model page directly:
```
WebFetch: https://huggingface.co/{org}/{model}
Prompt: Extract all benchmark scores and evaluation results from this model card
```

---

## Step 2: Search for Benchmark Variations

Model cards use inconsistent naming. Search for these variations:

### HLE (Humanity's Last Exam)
- `HLE`
- `hle`
- `Humanity's Last Exam`
- `HLE (Text Only)`
- `hle_text_only`

### GPQA
- `GPQA`
- `GPQA Diamond`
- `gpqa_diamond`
- `GPQA-Diamond`

### MMLU-Pro
- `MMLU-Pro`
- `MMLU Pro`
- `mmlu_pro`
- `MMLU-PRO`

### MMLU
- `MMLU`
- `mmlu`
- `Massive Multitask Language Understanding`

### GSM8K
- `GSM8K`
- `gsm8k`
- `GSM-8K`
- `Grade School Math`

### Other Common Benchmarks
| Benchmark | Variations |
|-----------|------------|
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

## Step 4: Extract and Normalize the Score

### Score Format Normalization

Scores may be presented as:
- **Percentages**: `85.2%` or `85.2` (when context implies %)
- **Decimals**: `0.852` (multiply by 100 for percentage)
- **Fractions**: `85.2/100`

**Important**: The `.eval_results/` format expects decimal values (0-1 scale) for most benchmarks, but some benchmarks use percentage scale. Check the source data to determine the correct scale.

### Extraction Regex Patterns

```python
import re

# Find score after benchmark name
patterns = [
    r'(?:HLE|hle)[^\d]*(\d+\.?\d*)',           # HLE: 12.3 or HLE 12.3%
    r'(?:GPQA|gpqa)[^\d]*(\d+\.?\d*)',         # GPQA variations
    r'(?:MMLU-Pro|mmlu.pro)[^\d]*(\d+\.?\d*)', # MMLU-Pro variations
]

# For table extraction
table_row = r'\|\s*(?:HLE|hle)[^\|]*\|\s*(\d+\.?\d*)'
```

---

## Step 5: Create the Evaluation Result

Once you have the score, format it for `.eval_results/`:

```yaml
# .eval_results/{benchmark}.yaml
- dataset:
    id: cais/hle           # Hub dataset ID (see mapping below)
    task_id: default       # Task variant if applicable
  value: 0.123             # Score as decimal (12.3% = 0.123)
  date: "2026-01-14"       # ISO date of extraction
  source:
    url: https://huggingface.co/{org}/{model}
    name: Model Card
```

### Dataset ID Mapping

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

## Example: Manual Extraction Workflow

```
1. Automated extraction failed:
   $ uv run scripts/evaluation_manager.py add-eval --benchmark HLE --repo-id "org/model"
   > Not found: HLE score not available

2. Fetch README manually:
   $ curl -s "https://huggingface.co/org/model/raw/main/README.md" | grep -i "hle\|humanity"

3. Found in unusual format:
   > "Scores on Humanity's Last Exam (text-only): 18.5%"

4. Create PR with manual value:
   $ uv run scripts/evaluation_manager.py add-eval \
       --benchmark HLE \
       --repo-id "org/model" \
       --value 18.5 \
       --create-pr
```

---

## Common Extraction Failures and Solutions

### Problem: Score in image/figure only
**Solution**: Check if there's a linked technical report or paper with tabular data

### Problem: Benchmark name differs significantly
**Solution**: Search for the underlying task name (e.g., "graduate-level science" for GPQA)

### Problem: Score aggregated with other metrics
**Solution**: Look for breakdown tables or supplementary materials

### Problem: Multiple scores for same benchmark (different settings)
**Solution**: Prefer "0-shot" or "standard" settings; note the configuration in source attribution

---

## Reporting Unextractable Scores

If manual extraction also fails, document why:
- Score genuinely not present in model card
- Score only in non-text format (images, PDFs without text layer)
- Benchmark not evaluated by model authors

This helps distinguish "not found by automation" from "truly not available".
