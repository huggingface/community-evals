# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "markdown-it-py>=3.0.0",
#     "python-dotenv>=1.2.1",
#     "pyyaml>=6.0.3",
#     "requests>=2.32.5",
# ]
# ///

"""
Manage evaluation results in Hugging Face model repositories.

This script provides methods for adding evaluation results to HuggingFace models:
1. Extract evaluation tables from model README files
2. Import evaluation scores from Artificial Analysis API

Evaluation results are stored in the new .eval_results/ format as documented at:
https://huggingface.co/docs/hub/eval-results

The new format stores results as YAML files in .eval_results/*.yaml with each entry
referencing a Hub Benchmark dataset ID.
"""

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple


def load_env() -> None:
    """Load .env if python-dotenv is available; keep help usable without it."""
    try:
        import dotenv  # type: ignore
    except ModuleNotFoundError:
        return
    dotenv.load_dotenv()


def require_markdown_it():
    try:
        from markdown_it import MarkdownIt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "markdown-it-py is required for table parsing. "
            "Install with `uv add markdown-it-py` or `pip install markdown-it-py`."
        ) from exc
    return MarkdownIt


def require_model_card():
    try:
        from huggingface_hub import ModelCard  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface-hub is required for model card operations. "
            "Install with `uv add huggingface_hub` or `pip install huggingface-hub`."
        ) from exc
    return ModelCard


def require_requests():
    try:
        import requests  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "requests is required for Artificial Analysis import. "
            "Install with `uv add requests` or `pip install requests`."
        ) from exc
    return requests


def require_yaml():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for YAML output. "
            "Install with `uv add pyyaml` or `pip install pyyaml`."
        ) from exc
    return yaml


def require_hf_api():
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface-hub is required for API operations. "
            "Install with `uv add huggingface_hub` or `pip install huggingface-hub`."
        ) from exc
    return HfApi


# ============================================================================
# Benchmark Mapping for .eval_results/ Format
# ============================================================================


def load_benchmark_mapping() -> Dict[str, Any]:
    """
    Load the benchmark-to-dataset mapping from metric_mapping.json.

    Returns:
        Dictionary mapping benchmark names to Hub dataset IDs and task IDs.
    """
    # Try to find the mapping file relative to this script
    script_dir = Path(__file__).parent
    mapping_file = script_dir.parent / "examples" / "metric_mapping.json"

    if not mapping_file.exists():
        # Fallback to a minimal built-in mapping
        return {
            "GPQA": {"dataset_id": "Idavidrein/gpqa", "task_id": "gpqa_diamond", "aliases": ["gpqa"]},
            "HLE": {"dataset_id": "cais/hle", "task_id": "default", "aliases": ["hle"]},
            "SimpleQA": {"dataset_id": "OpenEvals/SimpleQA", "task_id": "default", "aliases": ["simpleqa"]},
            "AIME": {"dataset_id": "OpenEvals/aime_24", "task_id": "default", "aliases": ["aime"]},
            "MMLU": {"dataset_id": "cais/mmlu", "task_id": "default", "aliases": ["mmlu"]},
            "GSM8K": {"dataset_id": "openai/gsm8k", "task_id": "default", "aliases": ["gsm8k"]},
            "BJJ-VQA": {"dataset_id": "couto/bjj-vqa", "task_id": "bjj_vqa", "aliases": ["bjj-vqa", "bjjvqa"]},
        }

    with open(mapping_file) as f:
        mapping = json.load(f)

    # Remove the comment field if present
    mapping.pop("_comment", None)
    return mapping


def find_benchmark_dataset(benchmark_name: str, mapping: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Find the Hub dataset ID for a benchmark name.

    Args:
        benchmark_name: Name of the benchmark (e.g., "MMLU", "gsm8k", "GPQA Diamond")
        mapping: Benchmark mapping dictionary

    Returns:
        Dictionary with dataset_id and task_id, or None if not found
    """
    # Clean markdown formatting first
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', benchmark_name)  # Remove links
    cleaned = re.sub(r'\*\*([^\*]+)\*\*', r'\1', cleaned)  # Remove bold
    cleaned = re.sub(r'\*([^\*]+)\*', r'\1', cleaned)  # Remove italic
    cleaned = cleaned.strip()

    # Normalize the benchmark name for matching
    normalized = cleaned.lower().replace(" ", "_").replace("-", "_")

    # Also create a version without parenthetical suffixes for fallback matching
    base_name = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned).strip()
    base_normalized = base_name.lower().replace(" ", "_").replace("-", "_")

    # Try exact match first (with cleaned name)
    if cleaned in mapping:
        entry = mapping[cleaned]
        return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    # Try case-insensitive exact match
    for key, entry in mapping.items():
        if key.lower() == cleaned.lower():
            return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    # Try matching against aliases (normalized)
    for key, entry in mapping.items():
        aliases = entry.get("aliases", [])
        normalized_aliases = [a.lower().replace(" ", "_").replace("-", "_") for a in aliases]
        if normalized in normalized_aliases:
            return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    # Try exact normalized key match (e.g., "mmlu_pro" matches "MMLU-Pro")
    for key, entry in mapping.items():
        key_normalized = key.lower().replace(" ", "_").replace("-", "_")
        if normalized == key_normalized:
            return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    # Fallback: try matching without parenthetical suffixes (e.g., "HLE (Text-only)" -> "HLE")
    if base_normalized != normalized:
        for key, entry in mapping.items():
            if key.lower() == base_name.lower():
                return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}
            key_normalized = key.lower().replace(" ", "_").replace("-", "_")
            if base_normalized == key_normalized:
                return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    return None


def convert_to_eval_results_format(
    metrics: List[Dict[str, Any]],
    source_url: Optional[str] = None,
    source_name: Optional[str] = None,
    source_user: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert extracted metrics to the new .eval_results/ YAML format.

    The new format stores evaluation results as YAML files in .eval_results/*.yaml
    with each entry referencing a Hub Benchmark dataset ID.

    Args:
        metrics: List of metric dictionaries with name, type, and value
        source_url: Optional source URL for attribution
        source_name: Optional source display name
        source_user: Optional HF username/org for attribution

    Returns:
        List of eval result entries in the new format
    """
    mapping = load_benchmark_mapping()
    results = []
    today = date.today().isoformat()

    for metric in metrics:
        benchmark_name = metric.get("name", "")
        value = metric.get("value")

        if value is None:
            continue

        # Find the dataset ID for this benchmark
        dataset_info = find_benchmark_dataset(benchmark_name, mapping)

        if not dataset_info:
            print(f"Warning: Could not find Hub dataset ID for benchmark '{benchmark_name}'. Skipping.")
            continue

        entry = {
            "dataset": {
                "id": dataset_info["dataset_id"],
            },
            "value": value,
            "date": today,
        }

        # Add task_id if not default
        if dataset_info.get("task_id") and dataset_info["task_id"] != "default":
            entry["dataset"]["task_id"] = dataset_info["task_id"]

        # Add source if provided
        if source_url:
            entry["source"] = {"url": source_url}
            if source_name:
                entry["source"]["name"] = source_name
            if source_user:
                entry["source"]["user"] = source_user

        results.append(entry)

    return results


def upload_eval_results(
    repo_id: str,
    results: List[Dict[str, Any]],
    filename: str = "evaluations.yaml",
    create_pr: bool = False,
    commit_message: Optional[str] = None,
) -> bool:
    """
    Upload evaluation results to the .eval_results/ folder in a model repository.

    Args:
        repo_id: Hugging Face repository ID
        results: List of eval result entries in the new format
        filename: Name of the YAML file (without path)
        create_pr: Whether to create a PR instead of direct push
        commit_message: Custom commit message

    Returns:
        True if successful, False otherwise
    """
    try:
        load_env()
        HfApi = require_hf_api()
        yaml = require_yaml()

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")

        api = HfApi(token=hf_token)

        # Generate YAML content
        yaml_content = yaml.dump(results, sort_keys=False, allow_unicode=True, default_flow_style=False)

        # Prepare file path
        file_path = f".eval_results/{filename}"

        # Prepare commit message
        if not commit_message:
            model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
            commit_message = f"Add evaluation results for {model_name}"

        # Build PR description with documentation links and images
        pr_description = """## Evaluation Results

This PR adds structured evaluation results using the new [`.eval_results/` format](https://huggingface.co/docs/hub/eval-results).

### What This Enables

- **Model Page**: Results appear on the model page with benchmark links
- **Leaderboards**: Scores are aggregated into benchmark dataset leaderboards
- **Verification**: Support for cryptographic verification of evaluation runs

![Model Evaluation Results](https://huggingface.co/huggingface/documentation-images/resolve/main/evaluation-results/eval-results-previw.png)

### Format Details

Results are stored as YAML in `.eval_results/` folder. See the [Eval Results Documentation](https://huggingface.co/docs/hub/eval-results) for the full specification.

---
*Generated by [community-evals](https://github.com/huggingface/community-evals)*"""

        # Upload file
        api.upload_file(
            path_or_fileobj=yaml_content.encode("utf-8"),
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            commit_description=pr_description,
            create_pr=create_pr,
        )

        action = "Pull request created" if create_pr else "Evaluation results uploaded"
        print(f"✓ {action} successfully for {repo_id}")
        print(f"  File: {file_path}")
        return True

    except Exception as e:
        print(f"Error uploading evaluation results: {e}")
        return False


# ============================================================================
# Method 1: Extract Evaluations from README
# ============================================================================


def extract_tables_from_markdown(markdown_content: str) -> List[str]:
    """Extract all markdown tables from content."""
    # Pattern to match markdown tables
    table_pattern = r"(\|[^\n]+\|(?:\r?\n\|[^\n]+\|)+)"
    tables = re.findall(table_pattern, markdown_content)
    return tables


def parse_markdown_table(table_str: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse a markdown table string into headers and rows.

    Returns:
        Tuple of (headers, data_rows)
    """
    lines = [line.strip() for line in table_str.strip().split("\n")]

    # Remove separator line (the one with dashes)
    lines = [line for line in lines if not re.match(r"^\|[\s\-:]+\|$", line)]

    if len(lines) < 2:
        return [], []

    # Parse header
    header = [cell.strip() for cell in lines[0].split("|")[1:-1]]

    # Parse data rows
    data_rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells:
            data_rows.append(cells)

    return header, data_rows


def is_evaluation_table(header: List[str], rows: List[List[str]]) -> bool:
    """Determine if a table contains evaluation results."""
    if not header or not rows:
        return False

    # Check if first column looks like benchmark names
    benchmark_keywords = [
        "benchmark", "task", "dataset", "eval", "test", "metric",
        "mmlu", "humaneval", "gsm", "hellaswag", "arc", "winogrande",
        "truthfulqa", "boolq", "piqa", "siqa"
    ]

    first_col = header[0].lower()
    has_benchmark_header = any(keyword in first_col for keyword in benchmark_keywords)

    # Check if there are numeric values in the table
    has_numeric_values = False
    for row in rows:
        for cell in row:
            try:
                float(cell.replace("%", "").replace(",", ""))
                has_numeric_values = True
                break
            except ValueError:
                continue
        if has_numeric_values:
            break

    return has_benchmark_header or has_numeric_values


def normalize_model_name(name: str) -> tuple[set[str], str]:
    """
    Normalize a model name for matching.

    Args:
        name: Model name to normalize

    Returns:
        Tuple of (token_set, normalized_string)
    """
    # Remove markdown formatting
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', name)  # Remove markdown links
    cleaned = re.sub(r'\*\*([^\*]+)\*\*', r'\1', cleaned)  # Remove bold
    cleaned = cleaned.strip()

    # Normalize and tokenize
    normalized = cleaned.lower().replace("-", " ").replace("_", " ")
    tokens = set(normalized.split())

    return tokens, normalized


def find_main_model_column(header: List[str], model_name: str) -> Optional[int]:
    """
    Identify the column index that corresponds to the main model.

    Only returns a column if there's an exact normalized match with the model name.
    This prevents extracting scores from training checkpoints or similar models.

    Args:
        header: Table column headers
        model_name: Model name from repo_id (e.g., "OLMo-3-32B-Think")

    Returns:
        Column index of the main model, or None if no exact match found
    """
    if not header or not model_name:
        return None

    # Normalize model name and extract tokens
    model_tokens, _ = normalize_model_name(model_name)

    # Find exact matches only
    for i, col_name in enumerate(header):
        if not col_name:
            continue

        # Skip first column (benchmark names)
        if i == 0:
            continue

        col_tokens, _ = normalize_model_name(col_name)

        # Check for exact token match
        if model_tokens == col_tokens:
            return i

    # No exact match found
    return None


def find_main_model_row(
    rows: List[List[str]], model_name: str
) -> tuple[Optional[int], List[str]]:
    """
    Identify the row index that corresponds to the main model in a transposed table.

    In transposed tables, each row represents a different model, with the first
    column containing the model name.

    Args:
        rows: Table data rows
        model_name: Model name from repo_id (e.g., "OLMo-3-32B")

    Returns:
        Tuple of (row_index, available_models)
        - row_index: Index of the main model, or None if no exact match found
        - available_models: List of all model names found in the table
    """
    if not rows or not model_name:
        return None, []

    model_tokens, _ = normalize_model_name(model_name)
    available_models = []

    for i, row in enumerate(rows):
        if not row or not row[0]:
            continue

        row_name = row[0].strip()

        # Skip separator/header rows
        if not row_name or row_name.startswith('---'):
            continue

        row_tokens, _ = normalize_model_name(row_name)

        # Collect all non-empty model names
        if row_tokens:
            available_models.append(row_name)

        # Check for exact token match
        if model_tokens == row_tokens:
            return i, available_models

    return None, available_models


def is_transposed_table(header: List[str], rows: List[List[str]]) -> bool:
    """
    Determine if a table is transposed (models as rows, benchmarks as columns).

    A table is considered transposed if:
    - The first column contains model-like names (not benchmark names)
    - Most other columns contain numeric values
    - Header row contains benchmark-like names

    Args:
        header: Table column headers
        rows: Table data rows

    Returns:
        True if table appears to be transposed, False otherwise
    """
    if not header or not rows or len(header) < 3:
        return False

    # Check if first column header suggests model names
    first_col = header[0].lower()
    model_indicators = ["model", "system", "llm", "name"]
    has_model_header = any(indicator in first_col for indicator in model_indicators)

    # Check if remaining headers look like benchmarks
    benchmark_keywords = [
        "mmlu", "humaneval", "gsm", "hellaswag", "arc", "winogrande",
        "eval", "score", "benchmark", "test", "math", "code", "mbpp",
        "truthfulqa", "boolq", "piqa", "siqa", "drop", "squad"
    ]

    benchmark_header_count = 0
    for col_name in header[1:]:
        col_lower = col_name.lower()
        if any(keyword in col_lower for keyword in benchmark_keywords):
            benchmark_header_count += 1

    has_benchmark_headers = benchmark_header_count >= 2

    # Check if data rows have numeric values in most columns (except first)
    numeric_count = 0
    total_cells = 0

    for row in rows[:5]:  # Check first 5 rows
        for cell in row[1:]:  # Skip first column
            total_cells += 1
            try:
                float(cell.replace("%", "").replace(",", "").strip())
                numeric_count += 1
            except (ValueError, AttributeError):
                continue

    has_numeric_data = total_cells > 0 and (numeric_count / total_cells) > 0.5

    return (has_model_header or has_benchmark_headers) and has_numeric_data


def extract_metrics_from_table(
    header: List[str],
    rows: List[List[str]],
    table_format: str = "auto",
    model_name: Optional[str] = None,
    model_column_index: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract metrics from parsed table data.

    Args:
        header: Table column headers
        rows: Table data rows
        table_format: "rows" (benchmarks as rows), "columns" (benchmarks as columns),
                     "transposed" (models as rows, benchmarks as columns), or "auto"
        model_name: Optional model name to identify the correct column/row

    Returns:
        List of metric dictionaries with name, type, and value
    """
    metrics = []

    if table_format == "auto":
        # First check if it's a transposed table (models as rows)
        if is_transposed_table(header, rows):
            table_format = "transposed"
        else:
            # Check if first column header is empty/generic (indicates benchmarks in rows)
            first_header = header[0].lower().strip() if header else ""
            is_first_col_benchmarks = not first_header or first_header in ["", "benchmark", "task", "dataset", "metric", "eval"]

            if is_first_col_benchmarks:
                table_format = "rows"
            else:
                # Heuristic: if first row has mostly numeric values, benchmarks are columns
                try:
                    numeric_count = sum(
                        1 for cell in rows[0] if cell and
                        re.match(r"^\d+\.?\d*%?$", cell.replace(",", "").strip())
                    )
                    table_format = "columns" if numeric_count > len(rows[0]) / 2 else "rows"
                except (IndexError, ValueError):
                    table_format = "rows"

    if table_format == "rows":
        # Benchmarks are in rows, scores in columns
        # Try to identify the main model column if model_name is provided
        target_column = model_column_index
        if target_column is None and model_name:
            target_column = find_main_model_column(header, model_name)

        for row in rows:
            if not row:
                continue

            benchmark_name = row[0].strip()
            if not benchmark_name:
                continue

            # If we identified a specific column, use it; otherwise use first numeric value
            if target_column is not None and target_column < len(row):
                try:
                    value_str = row[target_column].replace("%", "").replace(",", "").strip()
                    if value_str:
                        value = float(value_str)
                        metrics.append({
                            "name": benchmark_name,
                            "type": benchmark_name.lower().replace(" ", "_"),
                            "value": value
                        })
                except (ValueError, IndexError):
                    pass
            else:
                # Extract numeric values from remaining columns (original behavior)
                for i, cell in enumerate(row[1:], start=1):
                    try:
                        # Remove common suffixes and convert to float
                        value_str = cell.replace("%", "").replace(",", "").strip()
                        if not value_str:
                            continue

                        value = float(value_str)

                        # Determine metric name
                        metric_name = benchmark_name
                        if len(header) > i and header[i].lower() not in ["score", "value", "result"]:
                            metric_name = f"{benchmark_name} ({header[i]})"

                        metrics.append({
                            "name": metric_name,
                            "type": benchmark_name.lower().replace(" ", "_"),
                            "value": value
                        })
                        break  # Only take first numeric value per row
                    except (ValueError, IndexError):
                        continue

    elif table_format == "transposed":
        # Models are in rows (first column), benchmarks are in columns (header)
        # Find the row that matches the target model
        if not model_name:
            print("Warning: model_name required for transposed table format")
            return metrics

        target_row_idx, available_models = find_main_model_row(rows, model_name)

        if target_row_idx is None:
            print(f"\n⚠ Could not find model '{model_name}' in transposed table")
            if available_models:
                print("\nAvailable models in table:")
                for i, model in enumerate(available_models, 1):
                    print(f"  {i}. {model}")
                print("\nPlease select the correct model name from the list above.")
                print("You can specify it using the --model-name-override flag:")
                print(f'  --model-name-override "{available_models[0]}"')
            return metrics

        target_row = rows[target_row_idx]

        # Extract metrics from each column (skip first column which is model name)
        for i in range(1, len(header)):
            benchmark_name = header[i].strip()
            if not benchmark_name or i >= len(target_row):
                continue

            try:
                value_str = target_row[i].replace("%", "").replace(",", "").strip()
                if not value_str:
                    continue

                value = float(value_str)

                metrics.append({
                    "name": benchmark_name,
                    "type": benchmark_name.lower().replace(" ", "_").replace("-", "_"),
                    "value": value
                })
            except (ValueError, AttributeError):
                continue

    else:  # table_format == "columns"
        # Benchmarks are in columns
        if not rows:
            return metrics

        # Use first data row for values
        data_row = rows[0]

        for i, benchmark_name in enumerate(header):
            if not benchmark_name or i >= len(data_row):
                continue

            try:
                value_str = data_row[i].replace("%", "").replace(",", "").strip()
                if not value_str:
                    continue

                value = float(value_str)

                metrics.append({
                    "name": benchmark_name,
                    "type": benchmark_name.lower().replace(" ", "_"),
                    "value": value
                })
            except ValueError:
                continue

    return metrics


def extract_evaluations_from_readme(
    repo_id: str,
    task_type: str = "text-generation",
    dataset_name: str = "Benchmarks",
    dataset_type: str = "benchmark",
    model_name_override: Optional[str] = None,
    table_index: Optional[int] = None,
    model_column_index: Optional[int] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract evaluation results from a model's README.

    Args:
        repo_id: Hugging Face model repository ID
        task_type: Task type for model-index (e.g., "text-generation")
        dataset_name: Name for the benchmark dataset
        dataset_type: Type identifier for the dataset
        model_name_override: Override model name for matching (column header for comparison tables)
        table_index: 1-indexed table number from inspect-tables output

    Returns:
        Model-index formatted results or None if no evaluations found
    """
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)
        readme_content = card.content

        if not readme_content:
            print(f"No README content found for {repo_id}")
            return None

        # Extract model name from repo_id or use override
        if model_name_override:
            model_name = model_name_override
            print(f"Using model name override: '{model_name}'")
        else:
            model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Use markdown-it parser for accurate table extraction
        all_tables = extract_tables_with_parser(readme_content)

        if not all_tables:
            print(f"No tables found in README for {repo_id}")
            return None

        # If table_index specified, use that specific table
        if table_index is not None:
            if table_index < 1 or table_index > len(all_tables):
                print(f"Invalid table index {table_index}. Found {len(all_tables)} tables.")
                print("Run inspect-tables to see available tables.")
                return None
            tables_to_process = [all_tables[table_index - 1]]
        else:
            # Filter to evaluation tables only
            eval_tables = []
            for table in all_tables:
                header = table.get("headers", [])
                rows = table.get("rows", [])
                if is_evaluation_table(header, rows):
                    eval_tables.append(table)

            if len(eval_tables) > 1:
                print(f"\n⚠ Found {len(eval_tables)} evaluation tables.")
                print("Run inspect-tables first, then use --table to select one:")
                print(f'  uv run scripts/evaluation_manager.py inspect-tables --repo-id "{repo_id}"')
                return None
            elif len(eval_tables) == 0:
                print(f"No evaluation tables found in README for {repo_id}")
                return None

            tables_to_process = eval_tables

        # Extract metrics from selected table(s)
        all_metrics = []
        for table in tables_to_process:
            header = table.get("headers", [])
            rows = table.get("rows", [])
            metrics = extract_metrics_from_table(
                header,
                rows,
                model_name=model_name,
                model_column_index=model_column_index
            )
            all_metrics.extend(metrics)

        if not all_metrics:
            print(f"No metrics extracted from table")
            return None

        # Build model-index structure
        display_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        results = [{
            "task": {"type": task_type},
            "dataset": {
                "name": dataset_name,
                "type": dataset_type
            },
            "metrics": all_metrics,
            "source": {
                "name": "Model README",
                "url": f"https://huggingface.co/{repo_id}"
            }
        }]

        return results

    except Exception as e:
        print(f"Error extracting evaluations from README: {e}")
        return None


def extract_metrics_from_readme(
    repo_id: str,
    model_name_override: Optional[str] = None,
    table_index: Optional[int] = None,
    model_column_index: Optional[int] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract metrics from a model's README and return in internal format.

    This is a simplified version that returns just the metrics list,
    suitable for conversion to the new .eval_results/ format.

    Args:
        repo_id: Hugging Face model repository ID
        model_name_override: Override model name for matching
        table_index: 1-indexed table number from inspect-tables output
        model_column_index: Column index for model scores

    Returns:
        List of metric dictionaries with name, type, and value, or None if failed
    """
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)
        readme_content = card.content

        if not readme_content:
            print(f"No README content found for {repo_id}")
            return None

        # Extract model name from repo_id or use override
        if model_name_override:
            model_name = model_name_override
            print(f"Using model name override: '{model_name}'")
        else:
            model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Use markdown-it parser for accurate table extraction
        all_tables = extract_tables_with_parser(readme_content)

        if not all_tables:
            print(f"No tables found in README for {repo_id}")
            return None

        # If table_index specified, use that specific table
        if table_index is not None:
            if table_index < 1 or table_index > len(all_tables):
                print(f"Invalid table index {table_index}. Found {len(all_tables)} tables.")
                print("Run inspect-tables to see available tables.")
                return None
            tables_to_process = [all_tables[table_index - 1]]
        else:
            # Filter to evaluation tables only
            eval_tables = []
            for table in all_tables:
                header = table.get("headers", [])
                rows = table.get("rows", [])
                if is_evaluation_table(header, rows):
                    eval_tables.append(table)

            if len(eval_tables) > 1:
                print(f"\n⚠ Found {len(eval_tables)} evaluation tables.")
                print("Run inspect-tables first, then use --table to select one:")
                print(f'  uv run scripts/evaluation_manager.py inspect-tables --repo-id "{repo_id}"')
                return None
            elif len(eval_tables) == 0:
                print(f"No evaluation tables found in README for {repo_id}")
                return None

            tables_to_process = eval_tables

        # Extract metrics from selected table(s)
        all_metrics = []
        for table in tables_to_process:
            header = table.get("headers", [])
            rows = table.get("rows", [])
            metrics = extract_metrics_from_table(
                header,
                rows,
                model_name=model_name,
                model_column_index=model_column_index
            )
            all_metrics.extend(metrics)

        if not all_metrics:
            print(f"No metrics extracted from table")
            return None

        return all_metrics

    except Exception as e:
        print(f"Error extracting metrics from README: {e}")
        return None


# ============================================================================
# Table Inspection (using markdown-it-py for accurate parsing)
# ============================================================================


def extract_tables_with_parser(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Extract tables from markdown using markdown-it-py parser.
    Uses GFM (GitHub Flavored Markdown) which includes table support.
    """
    MarkdownIt = require_markdown_it()
    # Disable linkify to avoid optional dependency errors; not needed for table parsing.
    md = MarkdownIt("gfm-like", {"linkify": False})
    tokens = md.parse(markdown_content)

    tables = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == "table_open":
            table_data = {"headers": [], "rows": []}
            current_row = []
            in_header = False

            i += 1
            while i < len(tokens) and tokens[i].type != "table_close":
                t = tokens[i]
                if t.type == "thead_open":
                    in_header = True
                elif t.type == "thead_close":
                    in_header = False
                elif t.type == "tr_open":
                    current_row = []
                elif t.type == "tr_close":
                    if in_header:
                        table_data["headers"] = current_row
                    else:
                        table_data["rows"].append(current_row)
                    current_row = []
                elif t.type == "inline":
                    current_row.append(t.content.strip())
                i += 1

            if table_data["headers"] or table_data["rows"]:
                tables.append(table_data)

        i += 1

    return tables


def detect_table_format(table: Dict[str, Any], repo_id: str) -> Dict[str, Any]:
    """Analyze a table to detect its format and identify model columns."""
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    if not headers or not rows:
        return {"format": "unknown", "columns": headers, "model_columns": [], "row_count": 0, "sample_rows": []}

    first_header = headers[0].lower() if headers else ""
    is_first_col_benchmarks = not first_header or first_header in ["", "benchmark", "task", "dataset", "metric", "eval"]

    # Check for numeric columns
    numeric_columns = []
    for col_idx in range(1, len(headers)):
        numeric_count = 0
        for row in rows[:5]:
            if col_idx < len(row):
                try:
                    val = re.sub(r'\s*\([^)]*\)', '', row[col_idx])
                    float(val.replace("%", "").replace(",", "").strip())
                    numeric_count += 1
                except (ValueError, AttributeError):
                    pass
        if numeric_count > len(rows[:5]) / 2:
            numeric_columns.append(col_idx)

    # Determine format
    if is_first_col_benchmarks and len(numeric_columns) > 1:
        format_type = "comparison"
    elif is_first_col_benchmarks and len(numeric_columns) == 1:
        format_type = "simple"
    elif len(numeric_columns) > len(headers) / 2:
        format_type = "transposed"
    else:
        format_type = "unknown"

    # Find model columns
    model_columns = []
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    model_tokens, _ = normalize_model_name(model_name)

    for idx, header in enumerate(headers):
        if idx == 0 and is_first_col_benchmarks:
            continue
        if header:
            header_tokens, _ = normalize_model_name(header)
            is_match = model_tokens == header_tokens
            is_partial = model_tokens.issubset(header_tokens) or header_tokens.issubset(model_tokens)
            model_columns.append({
                "index": idx,
                "header": header,
                "is_exact_match": is_match,
                "is_partial_match": is_partial and not is_match
            })

    return {
        "format": format_type,
        "columns": headers,
        "model_columns": model_columns,
        "row_count": len(rows),
        "sample_rows": [row[0] for row in rows[:5] if row]
    }


def inspect_tables(repo_id: str) -> None:
    """Inspect and display all evaluation tables in a model's README."""
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)
        readme_content = card.content

        if not readme_content:
            print(f"No README content found for {repo_id}")
            return

        tables = extract_tables_with_parser(readme_content)

        if not tables:
            print(f"No tables found in README for {repo_id}")
            return

        print(f"\n{'='*70}")
        print(f"Tables found in README for: {repo_id}")
        print(f"{'='*70}")

        eval_table_count = 0
        for table in tables:
            analysis = detect_table_format(table, repo_id)

            if analysis["format"] == "unknown" and not analysis.get("sample_rows"):
                continue

            eval_table_count += 1
            print(f"\n## Table {eval_table_count}")
            print(f"   Format: {analysis['format']}")
            print(f"   Rows: {analysis['row_count']}")

            print(f"\n   Columns ({len(analysis['columns'])}):")
            for col_info in analysis.get("model_columns", []):
                idx = col_info["index"]
                header = col_info["header"]
                if col_info["is_exact_match"]:
                    print(f"      [{idx}] {header}  ✓ EXACT MATCH")
                elif col_info["is_partial_match"]:
                    print(f"      [{idx}] {header}  ~ partial match")
                else:
                    print(f"      [{idx}] {header}")

            if analysis.get("sample_rows"):
                print(f"\n   Sample rows (first column):")
                for row_val in analysis["sample_rows"][:5]:
                    print(f"      - {row_val}")

        if eval_table_count == 0:
            print("\nNo evaluation tables detected.")
        else:
            print("\nSuggested next step:")
            print(f'  uv run scripts/evaluation_manager.py extract-readme --repo-id "{repo_id}" --table <table-number> [--model-column-index <column-index>]')

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"Error inspecting tables: {e}")


# ============================================================================
# Pull Request Management
# ============================================================================


def get_open_prs(repo_id: str) -> List[Dict[str, Any]]:
    """
    Fetch open pull requests for a Hugging Face model repository.

    Args:
        repo_id: Hugging Face model repository ID (e.g., "allenai/Olmo-3-32B-Think")

    Returns:
        List of open PR dictionaries with num, title, author, and createdAt
    """
    requests = require_requests()
    url = f"https://huggingface.co/api/models/{repo_id}/discussions"

    try:
        response = requests.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()

        data = response.json()
        discussions = data.get("discussions", [])

        open_prs = [
            {
                "num": d["num"],
                "title": d["title"],
                "author": d["author"]["name"],
                "createdAt": d.get("createdAt", "unknown"),
            }
            for d in discussions
            if d.get("status") == "open" and d.get("isPullRequest")
        ]

        return open_prs

    except requests.RequestException as e:
        print(f"Error fetching PRs from Hugging Face: {e}")
        return []


def list_open_prs(repo_id: str) -> None:
    """Display open pull requests for a model repository."""
    prs = get_open_prs(repo_id)

    print(f"\n{'='*70}")
    print(f"Open Pull Requests for: {repo_id}")
    print(f"{'='*70}")

    if not prs:
        print("\nNo open pull requests found.")
    else:
        print(f"\nFound {len(prs)} open PR(s):\n")
        for pr in prs:
            print(f"  PR #{pr['num']} - {pr['title']}")
            print(f"     Author: {pr['author']}")
            print(f"     Created: {pr['createdAt']}")
            print(f"     URL: https://huggingface.co/{repo_id}/discussions/{pr['num']}")
            print()

    print(f"{'='*70}\n")


# ============================================================================
# Method 2: Import from Artificial Analysis
# ============================================================================


# ============================================================================
# Add Single Benchmark Evaluation
# ============================================================================


def lookup_benchmark_from_model_card(
    repo_id: str,
    benchmark_name: str,
) -> Optional[float]:
    """
    Look up a specific benchmark score from a model's README.

    Args:
        repo_id: HuggingFace repository ID
        benchmark_name: Name of the benchmark to find (e.g., "HLE", "GPQA")

    Returns:
        The benchmark score if found, None otherwise
    """
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)
        readme_content = card.content

        if not readme_content:
            return None

        # Get model name for column matching
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Extract all tables
        all_tables = extract_tables_with_parser(readme_content)

        if not all_tables:
            return None

        # Clean and normalize the target benchmark name for matching
        target_cleaned = re.sub(r'\*\*([^\*]+)\*\*', r'\1', benchmark_name)
        target_cleaned = target_cleaned.strip().lower()
        target_normalized = target_cleaned.replace(" ", "_").replace("-", "_")
        # Also create base version without parenthetical
        target_base = re.sub(r'\s*\([^)]*\)\s*$', '', target_cleaned).strip()

        # Search all tables for the benchmark
        for table in all_tables:
            header = table.get("headers", [])
            rows = table.get("rows", [])

            if not header or not rows:
                continue

            # Find model column by searching header for partial match
            model_col = None
            model_name_lower = model_name.lower()
            model_name_parts = set(model_name_lower.replace("-", " ").replace("_", " ").split())

            for col_idx, col_name in enumerate(header):
                col_lower = re.sub(r'<[^>]+>', '', col_name).lower().strip()  # Remove HTML tags
                col_parts = set(col_lower.replace("-", " ").replace("_", " ").split())

                # Check for significant overlap in name parts
                common = model_name_parts & col_parts
                if len(common) >= 1 and any(len(p) > 2 for p in common):
                    # Found a likely match
                    model_col = col_idx
                    break

            if model_col is None:
                continue

            # Search rows for the benchmark
            # Handle continuation rows (where benchmark name is in a previous row)
            current_benchmark = None
            for row in rows:
                if not row or len(row) <= model_col:
                    continue

                row_name = row[0].strip()
                if row_name:  # Non-empty first column = new benchmark
                    row_cleaned = re.sub(r'\*\*([^\*]+)\*\*', r'\1', row_name).strip().lower()
                    row_normalized = row_cleaned.replace(" ", "_").replace("-", "_")
                    row_base = re.sub(r'\s*\([^)]*\)\s*$', '', row_cleaned).strip()
                    current_benchmark = row_base
                # else: continuation row, use current_benchmark

                # Check if this row matches our target benchmark
                matches = False
                if row_name:
                    matches = (target_normalized == row_normalized or
                               target_cleaned == row_cleaned or
                               target_cleaned == row_base or
                               target_base == row_base)
                if not matches and current_benchmark:
                    matches = (target_cleaned == current_benchmark or
                               target_base == current_benchmark or
                               target_normalized == current_benchmark.replace(" ", "_").replace("-", "_"))

                if matches:
                    # Found the benchmark row, get the value
                    try:
                        value_str = row[model_col]
                        # Remove %, *, and other non-numeric chars
                        value_str = re.sub(r'[%*,]', '', value_str).strip()
                        if value_str and value_str != '-':
                            return float(value_str)
                    except (ValueError, IndexError):
                        continue

        return None

    except Exception as e:
        print(f"Error looking up benchmark from model card: {e}")
        return None


def lookup_benchmark_from_aa(
    repo_id: str,
    benchmark_name: str,
) -> Optional[float]:
    """
    Look up a specific benchmark score from Artificial Analysis API.

    Args:
        repo_id: HuggingFace repository ID
        benchmark_name: Name of the benchmark to find

    Returns:
        The benchmark score if found, None otherwise
    """
    try:
        load_env()
        requests = require_requests()

        AA_API_KEY = os.getenv("AA_API_KEY")
        if not AA_API_KEY:
            print("Warning: AA_API_KEY not set, cannot query Artificial Analysis")
            return None

        url = "https://artificialanalysis.ai/api/v2/data/llms/models"
        headers = {"x-api-key": AA_API_KEY}

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json().get("data", [])

        # Try to find the model by repo_id or model name
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        model_name_normalized = model_name.lower().replace("-", " ").replace("_", " ")

        target_model = None
        for model in data:
            aa_name = model.get("name", "").lower().replace("-", " ").replace("_", " ")
            aa_slug = model.get("slug", "").lower().replace("-", " ").replace("_", " ")
            if model_name_normalized in aa_name or model_name_normalized in aa_slug:
                target_model = model
                break

        if not target_model:
            return None

        # Look for the benchmark in evaluations
        evaluations = target_model.get("evaluations", {})
        benchmark_normalized = benchmark_name.lower().replace(" ", "_").replace("-", "_")

        for key, value in evaluations.items():
            key_normalized = key.lower().replace(" ", "_").replace("-", "_")
            if benchmark_normalized == key_normalized or benchmark_normalized in key_normalized:
                if value is not None:
                    return float(value)

        return None

    except Exception as e:
        print(f"Error looking up benchmark from Artificial Analysis: {e}")
        return None


def add_single_eval(
    repo_id: str,
    benchmark_name: str,
    source: str = "model_card",
    value: Optional[float] = None,
    create_pr: bool = False,
    apply: bool = False,
) -> bool:
    """
    Add a single benchmark evaluation result to a model repository.

    Args:
        repo_id: HuggingFace repository ID
        benchmark_name: Name of the benchmark (e.g., "HLE", "GPQA", "MMLU")
        source: Where to look up the score - "model_card", "aa", or "papers"
        value: Optional manual value (skips lookup if provided)
        create_pr: Create a PR instead of direct push
        apply: Actually upload the result (default is preview only)

    Returns:
        True if successful, False otherwise
    """
    yaml_mod = require_yaml()

    # Find the dataset mapping for this benchmark
    mapping = load_benchmark_mapping()
    dataset_info = find_benchmark_dataset(benchmark_name, mapping)

    if not dataset_info:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print("Check examples/metric_mapping.json for supported benchmarks")
        return False

    # Look up the value if not provided
    if value is None:
        print(f"Looking up {benchmark_name} score for {repo_id} from {source}...")

        if source == "model_card":
            value = lookup_benchmark_from_model_card(repo_id, benchmark_name)
        elif source == "aa":
            value = lookup_benchmark_from_aa(repo_id, benchmark_name)
        elif source == "papers":
            print("HuggingFace Papers lookup not yet implemented")
            return False
        else:
            print(f"Unknown source: {source}")
            return False

        if value is None:
            print(f"Could not find {benchmark_name} score in {source}")
            return False

        print(f"Found: {benchmark_name} = {value}")

    # Build the eval result entry
    today = date.today().isoformat()

    entry = {
        "dataset": {
            "id": dataset_info["dataset_id"],
        },
        "value": value,
        "date": today,
    }

    # Add task_id if not default
    if dataset_info.get("task_id") and dataset_info["task_id"] != "default":
        entry["dataset"]["task_id"] = dataset_info["task_id"]

    # Add source attribution based on lookup source
    if source == "model_card":
        entry["source"] = {
            "url": f"https://huggingface.co/{repo_id}",
            "name": "Model Card",
        }
    elif source == "aa":
        entry["source"] = {
            "url": "https://artificialanalysis.ai",
            "name": "Artificial Analysis",
        }
    elif source == "papers":
        entry["source"] = {
            "url": "https://huggingface.co/papers",
            "name": "HuggingFace Papers",
        }

    results = [entry]

    # Print preview
    print(f"\nGenerated .eval_results/ YAML for {benchmark_name}:")
    print(yaml_mod.dump(results, sort_keys=False, allow_unicode=True))

    # Upload if requested
    if apply or create_pr:
        # Use benchmark name as filename (sanitized)
        filename = benchmark_name.lower().replace(" ", "_").replace("-", "_") + ".yaml"
        return upload_eval_results(
            repo_id=repo_id,
            results=results,
            filename=filename,
            create_pr=create_pr,
            commit_message=f"Add {benchmark_name} evaluation result",
        )

    return True


def get_aa_model_data(creator_slug: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch model evaluation data from Artificial Analysis API.

    Args:
        creator_slug: Creator identifier (e.g., "anthropic", "openai")
        model_name: Model slug/identifier

    Returns:
        Model data dictionary or None if not found
    """
    load_env()
    AA_API_KEY = os.getenv("AA_API_KEY")
    if not AA_API_KEY:
        raise ValueError("AA_API_KEY environment variable is not set")

    url = "https://artificialanalysis.ai/api/v2/data/llms/models"
    headers = {"x-api-key": AA_API_KEY}

    requests = require_requests()

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json().get("data", [])

        for model in data:
            creator = model.get("model_creator", {})
            if creator.get("slug") == creator_slug and model.get("slug") == model_name:
                return model

        print(f"Model {creator_slug}/{model_name} not found in Artificial Analysis")
        return None

    except requests.RequestException as e:
        print(f"Error fetching data from Artificial Analysis: {e}")
        return None


def aa_data_to_model_index(
    model_data: Dict[str, Any],
    dataset_name: str = "Artificial Analysis Benchmarks",
    dataset_type: str = "artificial_analysis",
    task_type: str = "evaluation"
) -> List[Dict[str, Any]]:
    """
    Convert Artificial Analysis model data to model-index format.

    Args:
        model_data: Raw model data from AA API
        dataset_name: Dataset name for model-index
        dataset_type: Dataset type identifier
        task_type: Task type for model-index

    Returns:
        Model-index formatted results
    """
    model_name = model_data.get("name", model_data.get("slug", "unknown-model"))
    evaluations = model_data.get("evaluations", {})

    if not evaluations:
        print(f"No evaluations found for model {model_name}")
        return []

    metrics = []
    for key, value in evaluations.items():
        if value is not None:
            metrics.append({
                "name": key.replace("_", " ").title(),
                "type": key,
                "value": value
            })

    results = [{
        "task": {"type": task_type},
        "dataset": {
            "name": dataset_name,
            "type": dataset_type
        },
        "metrics": metrics,
        "source": {
            "name": "Artificial Analysis API",
            "url": "https://artificialanalysis.ai"
        }
    }]

    return results


def import_aa_evaluations(
    creator_slug: str,
    model_name: str,
    repo_id: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Import evaluation results from Artificial Analysis for a model.

    Args:
        creator_slug: Creator identifier in AA
        model_name: Model identifier in AA
        repo_id: Hugging Face repository ID to update

    Returns:
        Model-index formatted results or None if import fails
    """
    model_data = get_aa_model_data(creator_slug, model_name)

    if not model_data:
        return None

    results = aa_data_to_model_index(model_data)
    return results


# ============================================================================
# Model Card Update Functions
# ============================================================================


def update_model_card_with_evaluations(
    repo_id: str,
    results: List[Dict[str, Any]],
    create_pr: bool = False,
    commit_message: Optional[str] = None
) -> bool:
    """
    Update a model card with evaluation results.

    Args:
        repo_id: Hugging Face repository ID
        results: Model-index formatted results
        create_pr: Whether to create a PR instead of direct push
        commit_message: Custom commit message

    Returns:
        True if successful, False otherwise
    """
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")

        # Load existing card
        card = ModelCard.load(repo_id, token=hf_token)

        # Get model name
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Create or update model-index
        model_index = [{
            "name": model_name,
            "results": results
        }]

        # Merge with existing model-index if present
        if "model-index" in card.data:
            existing = card.data["model-index"]
            if isinstance(existing, list) and existing:
                # Keep existing name if present
                if "name" in existing[0]:
                    model_index[0]["name"] = existing[0]["name"]

                # Merge results
                existing_results = existing[0].get("results", [])
                model_index[0]["results"].extend(existing_results)

        card.data["model-index"] = model_index

        # Prepare commit message
        if not commit_message:
            commit_message = f"Add evaluation results to {model_name}"

        commit_description = """## Evaluation Results

This PR adds structured evaluation results to the model card using the model-index specification.

### Note

Consider migrating to the new [`.eval_results/` format](https://huggingface.co/docs/hub/eval-results) which enables:
- Results appearing on benchmark leaderboards
- Community contributions via PRs
- Verification of evaluation runs

![Model Evaluation Results](https://huggingface.co/huggingface/documentation-images/resolve/main/evaluation-results/eval-results-previw.png)

---
*Generated by [community-evals](https://github.com/huggingface/community-evals)*"""

        # Push update
        card.push_to_hub(
            repo_id,
            token=hf_token,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr
        )

        action = "Pull request created" if create_pr else "Model card updated"
        print(f"✓ {action} successfully for {repo_id}")
        return True

    except Exception as e:
        print(f"Error updating model card: {e}")
        return False


def show_evaluations(repo_id: str) -> None:
    """Display current evaluations in a model card."""
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)

        if "model-index" not in card.data:
            print(f"No model-index found in {repo_id}")
            return

        model_index = card.data["model-index"]

        print(f"\nEvaluations for {repo_id}:")
        print("=" * 60)

        for model_entry in model_index:
            model_name = model_entry.get("name", "Unknown")
            print(f"\nModel: {model_name}")

            results = model_entry.get("results", [])
            for i, result in enumerate(results, 1):
                print(f"\n  Result Set {i}:")

                task = result.get("task", {})
                print(f"    Task: {task.get('type', 'unknown')}")

                dataset = result.get("dataset", {})
                print(f"    Dataset: {dataset.get('name', 'unknown')}")

                metrics = result.get("metrics", [])
                print(f"    Metrics ({len(metrics)}):")
                for metric in metrics:
                    name = metric.get("name", "Unknown")
                    value = metric.get("value", "N/A")
                    print(f"      - {name}: {value}")

                source = result.get("source", {})
                if source:
                    print(f"    Source: {source.get('name', 'Unknown')}")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"Error showing evaluations: {e}")


def validate_model_index(repo_id: str) -> bool:
    """Validate model-index format in a model card."""
    try:
        load_env()
        ModelCard = require_model_card()
        hf_token = os.getenv("HF_TOKEN")
        card = ModelCard.load(repo_id, token=hf_token)

        if "model-index" not in card.data:
            print(f"✗ No model-index found in {repo_id}")
            return False

        model_index = card.data["model-index"]

        if not isinstance(model_index, list):
            print("✗ model-index must be a list")
            return False

        for i, entry in enumerate(model_index):
            if "name" not in entry:
                print(f"✗ Entry {i} missing 'name' field")
                return False

            if "results" not in entry:
                print(f"✗ Entry {i} missing 'results' field")
                return False

            for j, result in enumerate(entry["results"]):
                if "task" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'task' field")
                    return False

                if "dataset" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'dataset' field")
                    return False

                if "metrics" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'metrics' field")
                    return False

        print(f"✓ Model-index format is valid for {repo_id}")
        return True

    except Exception as e:
        print(f"Error validating model-index: {e}")
        return False


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Manage evaluation results in Hugging Face model repositories.\n\n"
            "Outputs results in the new .eval_results/ format (https://huggingface.co/docs/hub/eval-results).\n"
            "Use standard Python or `uv run scripts/evaluation_manager.py ...` "
            "to auto-resolve dependencies from the PEP 723 header."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=dedent(
            """\
            Typical workflows:
              - Inspect tables first:
                  uv run scripts/evaluation_manager.py inspect-tables --repo-id <model>
              - Extract from README (prints YAML by default):
                  uv run scripts/evaluation_manager.py extract-readme --repo-id <model> --table N
              - Apply changes (uploads to .eval_results/):
                  uv run scripts/evaluation_manager.py extract-readme --repo-id <model> --table N --apply
              - Import from Artificial Analysis:
                  AA_API_KEY=... uv run scripts/evaluation_manager.py import-aa --creator-slug org --model-name slug --repo-id <model>

            Tips:
              - YAML is printed by default; use --apply or --create-pr to write changes to .eval_results/.
              - Set HF_TOKEN (and AA_API_KEY for import-aa); .env is loaded automatically if python-dotenv is installed.
              - When multiple tables exist, run inspect-tables then select with --table N.
              - Results are stored in .eval_results/*.yaml and appear on model pages and benchmark leaderboards.
            """
        ),
    )
    parser.add_argument("--version", action="version", version="evaluation_manager 2.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract from README command
    extract_parser = subparsers.add_parser(
        "extract-readme",
        help="Extract evaluation tables from model README",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Parse README tables and output to .eval_results/ format. Default behavior prints YAML; use --apply/--create-pr to write changes.",
        epilog=dedent(
            """\
            Examples:
              uv run scripts/evaluation_manager.py extract-readme --repo-id username/model
              uv run scripts/evaluation_manager.py extract-readme --repo-id username/model --table 2 --model-column-index 3
              uv run scripts/evaluation_manager.py extract-readme --repo-id username/model --table 2 --model-name-override \"**Model 7B**\"
              uv run scripts/evaluation_manager.py extract-readme --repo-id username/model --table 2 --create-pr

            Apply changes:
              - Default: prints YAML to stdout (no writes).
              - Add --apply to push directly to .eval_results/, or --create-pr to open a PR.
            Model selection:
              - Preferred: --model-column-index <header index shown by inspect-tables>
              - If using --model-name-override, copy the column header text exactly.
            Output format:
              - Results are output in the new .eval_results/ format
              - Each benchmark maps to a Hub Benchmark dataset ID
              - See https://huggingface.co/docs/hub/eval-results for details
            """
        ),
    )
    extract_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")
    extract_parser.add_argument("--table", type=int, help="Table number (1-indexed, from inspect-tables output)")
    extract_parser.add_argument("--model-column-index", type=int, help="Preferred: column index from inspect-tables output (exact selection)")
    extract_parser.add_argument("--model-name-override", type=str, help="Exact column header/model name for comparison/transpose tables (when index is not used)")
    extract_parser.add_argument("--source-url", type=str, help="Source URL for attribution (e.g., evaluation logs, paper)")
    extract_parser.add_argument("--source-name", type=str, help="Source display name (e.g., 'Eval traces', 'Paper')")
    extract_parser.add_argument("--source-user", type=str, help="HF username/org for source attribution")
    extract_parser.add_argument("--filename", type=str, default="readme_evals.yaml", help="Output filename in .eval_results/ (default: readme_evals.yaml)")
    extract_parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    extract_parser.add_argument("--apply", action="store_true", help="Apply changes (default is to print YAML only)")
    extract_parser.add_argument("--dry-run", action="store_true", help="Preview YAML without updating (default)")

    # Import from AA command
    aa_parser = subparsers.add_parser(
        "import-aa",
        help="Import evaluation scores from Artificial Analysis",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Fetch scores from Artificial Analysis API and write them to .eval_results/.",
        epilog=dedent(
            """\
            Examples:
              AA_API_KEY=... uv run scripts/evaluation_manager.py import-aa --creator-slug anthropic --model-name claude-sonnet-4 --repo-id username/model
              uv run scripts/evaluation_manager.py import-aa --creator-slug openai --model-name gpt-4o --repo-id username/model --create-pr

            Requires: AA_API_KEY in env (or .env if python-dotenv installed).
            Output: Results are written to .eval_results/artificial_analysis.yaml
            """
        ),
    )
    aa_parser.add_argument("--creator-slug", type=str, required=True, help="AA creator slug")
    aa_parser.add_argument("--model-name", type=str, required=True, help="AA model name")
    aa_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")
    aa_parser.add_argument("--filename", type=str, default="artificial_analysis.yaml", help="Output filename in .eval_results/ (default: artificial_analysis.yaml)")
    aa_parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    aa_parser.add_argument("--apply", action="store_true", help="Apply changes (default is to print YAML only)")

    # Show evaluations command
    show_parser = subparsers.add_parser(
        "show",
        help="Display current evaluations in model card",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Print model-index content from the model card (requires HF_TOKEN for private repos).",
    )
    show_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate model-index format",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Schema sanity check for model-index section of the card.",
    )
    validate_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    # Inspect tables command
    inspect_parser = subparsers.add_parser(
        "inspect-tables",
        help="Inspect tables in README → outputs suggested extract-readme command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. inspect-tables     → see table structure, columns, and table numbers
  2. extract-readme     → run with --table N (from step 1); YAML prints by default
  3. apply changes      → rerun extract-readme with --apply or --create-pr

Reminder:
  - Preferred: use --model-column-index <index>. If needed, use --model-name-override with the exact column header text.
"""
    )
    inspect_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    # Add single eval command
    add_eval_parser = subparsers.add_parser(
        "add-eval",
        help="Add a single benchmark evaluation to a model",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Look up and add a specific benchmark score to .eval_results/.",
        epilog=dedent(
            """\
            Examples:
              # Look up HLE score from model card and preview
              uv run scripts/evaluation_manager.py add-eval --benchmark HLE --repo-id "moonshotai/Kimi-K2-Thinking"

              # Look up from Artificial Analysis
              uv run scripts/evaluation_manager.py add-eval --benchmark MMLU --repo-id "model/name" --source aa

              # Provide value manually
              uv run scripts/evaluation_manager.py add-eval --benchmark GPQA --repo-id "model/name" --value 84.5

              # Apply changes (upload to repo)
              uv run scripts/evaluation_manager.py add-eval --benchmark HLE --repo-id "model/name" --apply

            Sources:
              - model_card: Extract from the model's README (default)
              - aa: Query Artificial Analysis API (requires AA_API_KEY)
              - papers: Query HuggingFace Papers (not yet implemented)
            """
        ),
    )
    add_eval_parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (e.g., HLE, GPQA, MMLU)")
    add_eval_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")
    add_eval_parser.add_argument("--source", type=str, default="model_card", choices=["model_card", "aa", "papers"], help="Where to look up the score (default: model_card)")
    add_eval_parser.add_argument("--value", type=float, help="Manually provide the score (skips lookup)")
    add_eval_parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    add_eval_parser.add_argument("--apply", action="store_true", help="Apply changes (default is preview only)")

    # Get PRs command
    prs_parser = subparsers.add_parser(
        "get-prs",
        help="List open pull requests for a model repository",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Check for existing open PRs before creating new ones to avoid duplicates.",
        epilog=dedent(
            """\
            Examples:
              uv run scripts/evaluation_manager.py get-prs --repo-id "allenai/Olmo-3-32B-Think"

            IMPORTANT: Always run this before using --create-pr to avoid duplicate PRs.
            """
        ),
    )
    prs_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # Execute command
        if args.command == "extract-readme":
            # Extract metrics from README (internal format with name/type/value)
            metrics = extract_metrics_from_readme(
                repo_id=args.repo_id,
                model_name_override=args.model_name_override,
                table_index=args.table,
                model_column_index=args.model_column_index
            )

            if not metrics:
                print("No evaluations extracted")
                return

            # Convert to new .eval_results/ format
            source_url = getattr(args, 'source_url', None)
            if not source_url:
                source_url = f"https://huggingface.co/{args.repo_id}"

            eval_results = convert_to_eval_results_format(
                metrics=metrics,
                source_url=source_url,
                source_name=getattr(args, 'source_name', "Model README"),
                source_user=getattr(args, 'source_user', None),
            )

            if not eval_results:
                print("No benchmarks could be mapped to Hub dataset IDs")
                print("Check that benchmark names match entries in metric_mapping.json")
                return

            apply_changes = args.apply or args.create_pr

            # Default behavior: print YAML (dry-run)
            yaml = require_yaml()
            print("\nExtracted evaluations (.eval_results/ format):")
            print(yaml.dump(eval_results, sort_keys=False, allow_unicode=True))

            if apply_changes:
                if args.model_name_override and args.model_column_index is not None:
                    print("Note: --model-column-index takes precedence over --model-name-override.")
                upload_eval_results(
                    repo_id=args.repo_id,
                    results=eval_results,
                    filename=args.filename,
                    create_pr=args.create_pr,
                    commit_message="Extract evaluation results from README"
                )

        elif args.command == "import-aa":
            # Get raw AA data
            model_data = get_aa_model_data(args.creator_slug, args.model_name)

            if not model_data:
                print("No model data found in Artificial Analysis")
                return

            # Extract metrics from AA data
            evaluations = model_data.get("evaluations", {})
            if not evaluations:
                print(f"No evaluations found for model {args.model_name}")
                return

            metrics = []
            for key, value in evaluations.items():
                if value is not None:
                    metrics.append({
                        "name": key.replace("_", " ").title(),
                        "type": key,
                        "value": value
                    })

            # Convert to new .eval_results/ format
            eval_results = convert_to_eval_results_format(
                metrics=metrics,
                source_url="https://artificialanalysis.ai",
                source_name="Artificial Analysis",
            )

            if not eval_results:
                print("No benchmarks could be mapped to Hub dataset IDs")
                return

            apply_changes = args.apply or args.create_pr

            # Default behavior: print YAML
            yaml = require_yaml()
            print("\nImported evaluations (.eval_results/ format):")
            print(yaml.dump(eval_results, sort_keys=False, allow_unicode=True))

            if apply_changes:
                upload_eval_results(
                    repo_id=args.repo_id,
                    results=eval_results,
                    filename=args.filename,
                    create_pr=args.create_pr,
                    commit_message=f"Add Artificial Analysis evaluations for {args.model_name}"
                )

        elif args.command == "show":
            show_evaluations(args.repo_id)

        elif args.command == "validate":
            validate_model_index(args.repo_id)

        elif args.command == "inspect-tables":
            inspect_tables(args.repo_id)

        elif args.command == "add-eval":
            add_single_eval(
                repo_id=args.repo_id,
                benchmark_name=args.benchmark,
                source=args.source,
                value=args.value,
                create_pr=args.create_pr,
                apply=args.apply,
            )

        elif args.command == "get-prs":
            list_open_prs(args.repo_id)
    except ModuleNotFoundError as exc:
        # Surface dependency hints cleanly when user only needs help output
        print(exc)
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
