# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "markdown-it-py>=3.0.0",
#     "python-dotenv>=1.0.0",
#     "pyyaml>=6.0.0",
# ]
# ///
"""
Extract evaluation results from model card README tables.

Usage:
  # Inspect tables (dry run)
  uv run scripts/extract_model_card.py --repo-id "org/model" --inspect

  # Extract from specific table (dry run - prints YAML)
  uv run scripts/extract_model_card.py --repo-id "org/model" --table 1

  # Extract and create PR
  uv run scripts/extract_model_card.py --repo-id "org/model" --table 1 --create-pr

  # Extract specific benchmark
  uv run scripts/extract_model_card.py --repo-id "org/model" --benchmark HLE
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


def load_env() -> None:
    try:
        import dotenv
        dotenv.load_dotenv()
    except ModuleNotFoundError:
        pass


def load_benchmark_mapping() -> dict[str, Any]:
    script_dir = Path(__file__).parent
    mapping_file = script_dir.parent / "examples" / "metric_mapping.json"

    if not mapping_file.exists():
        return {
            "GPQA": {"dataset_id": "Idavidrein/gpqa", "task_id": "gpqa_diamond", "aliases": ["gpqa"]},
            "HLE": {"dataset_id": "cais/hle", "task_id": "default", "aliases": ["hle"]},
            "SimpleQA": {"dataset_id": "OpenEvals/SimpleQA", "task_id": "default", "aliases": ["simpleqa"]},
            "MMLU": {"dataset_id": "cais/mmlu", "task_id": "default", "aliases": ["mmlu"]},
            "GSM8K": {"dataset_id": "openai/gsm8k", "task_id": "default", "aliases": ["gsm8k"]},
        }

    with open(mapping_file) as f:
        mapping = json.load(f)
    mapping.pop("_comment", None)
    return mapping


def find_benchmark_dataset(benchmark_name: str, mapping: dict[str, Any]) -> dict[str, str] | None:
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', benchmark_name)
    cleaned = re.sub(r'\*\*([^\*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*([^\*]+)\*', r'\1', cleaned)
    cleaned = cleaned.strip()

    normalized = cleaned.lower().replace(" ", "_").replace("-", "_")
    base_name = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned).strip()
    base_normalized = base_name.lower().replace(" ", "_").replace("-", "_")

    if cleaned in mapping:
        entry = mapping[cleaned]
        return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    for key, entry in mapping.items():
        if key.lower() == cleaned.lower():
            return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    for key, entry in mapping.items():
        aliases = entry.get("aliases", [])
        normalized_aliases = [a.lower().replace(" ", "_").replace("-", "_") for a in aliases]
        if normalized in normalized_aliases:
            return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    for key, entry in mapping.items():
        key_normalized = key.lower().replace(" ", "_").replace("-", "_")
        if normalized == key_normalized:
            return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    if base_normalized != normalized:
        for key, entry in mapping.items():
            if key.lower() == base_name.lower():
                return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}
            key_normalized = key.lower().replace(" ", "_").replace("-", "_")
            if base_normalized == key_normalized:
                return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    return None


def extract_tables_with_parser(markdown_content: str) -> list[dict[str, Any]]:
    from markdown_it import MarkdownIt
    md = MarkdownIt("gfm-like", {"linkify": False})
    tokens = md.parse(markdown_content)

    tables = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == "table_open":
            table_data: dict[str, Any] = {"headers": [], "rows": []}
            current_row: list[str] = []
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


def is_evaluation_table(header: list[str], rows: list[list[str]]) -> bool:
    if not header or not rows:
        return False

    benchmark_keywords = [
        "benchmark", "task", "dataset", "eval", "test", "metric",
        "mmlu", "humaneval", "gsm", "hellaswag", "arc", "winogrande",
        "truthfulqa", "boolq", "piqa", "siqa", "gpqa", "hle"
    ]

    first_col = header[0].lower()
    has_benchmark_header = any(keyword in first_col for keyword in benchmark_keywords)

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
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', name)
    cleaned = re.sub(r'\*\*([^\*]+)\*\*', r'\1', cleaned)
    cleaned = cleaned.strip()

    normalized = cleaned.lower().replace("-", " ").replace("_", " ")
    tokens = set(normalized.split())

    return tokens, normalized


def extract_metrics_from_table(
    header: list[str],
    rows: list[list[str]],
    model_name: str | None = None,
    model_column_index: int | None = None
) -> list[dict[str, Any]]:
    metrics = []

    target_column = model_column_index
    if target_column is None and model_name:
        model_tokens, _ = normalize_model_name(model_name)
        for i, col_name in enumerate(header):
            if not col_name or i == 0:
                continue
            col_tokens, _ = normalize_model_name(col_name)
            if model_tokens == col_tokens:
                target_column = i
                break

    for row in rows:
        if not row:
            continue

        benchmark_name = row[0].strip()
        if not benchmark_name:
            continue

        if target_column is not None and target_column < len(row):
            try:
                value_str = row[target_column].replace("%", "").replace(",", "").strip()
                if value_str and value_str != '-':
                    value = float(value_str)
                    metrics.append({
                        "name": benchmark_name,
                        "type": benchmark_name.lower().replace(" ", "_"),
                        "value": value
                    })
            except (ValueError, IndexError):
                pass
        else:
            for i, cell in enumerate(row[1:], start=1):
                try:
                    value_str = cell.replace("%", "").replace(",", "").strip()
                    if not value_str:
                        continue
                    value = float(value_str)
                    metrics.append({
                        "name": benchmark_name,
                        "type": benchmark_name.lower().replace(" ", "_"),
                        "value": value
                    })
                    break
                except (ValueError, IndexError):
                    continue

    return metrics


def convert_to_eval_results_format(
    metrics: list[dict[str, Any]],
    source_url: str | None = None,
    source_name: str | None = None,
    source_user: str | None = None,
) -> list[dict[str, Any]]:
    mapping = load_benchmark_mapping()
    results = []
    today = date.today().isoformat()

    for metric in metrics:
        benchmark_name = metric.get("name", "")
        value = metric.get("value")

        if value is None:
            continue

        dataset_info = find_benchmark_dataset(benchmark_name, mapping)
        if not dataset_info:
            print(f"Warning: Could not find Hub dataset ID for benchmark '{benchmark_name}'. Skipping.", file=sys.stderr)
            continue

        entry: dict[str, Any] = {
            "dataset": {"id": dataset_info["dataset_id"]},
            "value": value,
            "date": today,
        }

        if dataset_info.get("task_id") and dataset_info["task_id"] != "default":
            entry["dataset"]["task_id"] = dataset_info["task_id"]

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
    results: list[dict[str, Any]],
    filename: str = "evaluations.yaml",
    create_pr: bool = False,
    commit_message: str | None = None,
) -> bool:
    import yaml
    from huggingface_hub import HfApi

    load_env()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is not set", file=sys.stderr)
        return False

    api = HfApi(token=hf_token)
    yaml_content = yaml.dump(results, sort_keys=False, allow_unicode=True, default_flow_style=False)
    file_path = f".eval_results/{filename}"

    if not commit_message:
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        commit_message = f"Add evaluation results for {model_name}"

    pr_description = """## Evaluation Results

This PR adds structured evaluation results using the new [`.eval_results/` format](https://huggingface.co/docs/hub/eval-results).

### What This Enables

- **Model Page**: Results appear on the model page with benchmark links
- **Leaderboards**: Scores are aggregated into benchmark dataset leaderboards
- **Verification**: Support for cryptographic verification of evaluation runs

---
*Generated by [community-evals](https://github.com/huggingface/community-evals)*"""

    try:
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
        print(f"Error uploading evaluation results: {e}", file=sys.stderr)
        return False


def inspect_tables(repo_id: str) -> None:
    from huggingface_hub import ModelCard

    load_env()
    hf_token = os.getenv("HF_TOKEN")

    try:
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
            header = table.get("headers", [])
            rows = table.get("rows", [])

            if not is_evaluation_table(header, rows):
                continue

            eval_table_count += 1
            print(f"\n## Table {eval_table_count}")
            print(f"   Headers: {header}")
            print(f"   Rows: {len(rows)}")

            if rows:
                print(f"   Sample rows (first column):")
                for row in rows[:5]:
                    if row:
                        print(f"      - {row[0]}")

        if eval_table_count == 0:
            print("\nNo evaluation tables detected.")
        else:
            print("\nSuggested next step:")
            print(f'  uv run scripts/extract_model_card.py --repo-id "{repo_id}" --table <N>')

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"Error inspecting tables: {e}", file=sys.stderr)


def extract_from_readme(
    repo_id: str,
    table_index: int | None = None,
    model_column_index: int | None = None,
    benchmark_name: str | None = None,
) -> list[dict[str, Any]] | None:
    from huggingface_hub import ModelCard

    load_env()
    hf_token = os.getenv("HF_TOKEN")

    try:
        card = ModelCard.load(repo_id, token=hf_token)
        readme_content = card.content

        if not readme_content:
            print(f"No README content found for {repo_id}", file=sys.stderr)
            return None

        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        all_tables = extract_tables_with_parser(readme_content)

        if not all_tables:
            print(f"No tables found in README for {repo_id}", file=sys.stderr)
            return None

        eval_tables = [t for t in all_tables if is_evaluation_table(t.get("headers", []), t.get("rows", []))]

        if table_index is not None:
            if table_index < 1 or table_index > len(eval_tables):
                print(f"Invalid table index {table_index}. Found {len(eval_tables)} evaluation tables.", file=sys.stderr)
                return None
            tables_to_process = [eval_tables[table_index - 1]]
        else:
            if len(eval_tables) > 1:
                print(f"\n⚠ Found {len(eval_tables)} evaluation tables.", file=sys.stderr)
                print("Run with --inspect first, then use --table to select one:", file=sys.stderr)
                print(f'  uv run scripts/extract_model_card.py --repo-id "{repo_id}" --inspect', file=sys.stderr)
                return None
            elif len(eval_tables) == 0:
                print(f"No evaluation tables found in README for {repo_id}", file=sys.stderr)
                return None
            tables_to_process = eval_tables

        all_metrics = []
        for table in tables_to_process:
            header = table.get("headers", [])
            rows = table.get("rows", [])
            metrics = extract_metrics_from_table(header, rows, model_name=model_name, model_column_index=model_column_index)
            all_metrics.extend(metrics)

        if benchmark_name:
            target_normalized = benchmark_name.lower().replace(" ", "_").replace("-", "_")
            all_metrics = [m for m in all_metrics if m.get("type", "").lower().replace(" ", "_").replace("-", "_") == target_normalized or m.get("name", "").lower() == benchmark_name.lower()]

        if not all_metrics:
            print(f"No metrics extracted from table", file=sys.stderr)
            return None

        return all_metrics

    except Exception as e:
        print(f"Error extracting from README: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract evaluation results from model card README tables.",
    )
    parser.add_argument("--repo-id", required=True, help="HuggingFace repository ID")
    parser.add_argument("--inspect", action="store_true", help="Inspect tables in README (dry run)")
    parser.add_argument("--table", type=int, help="Table number (1-indexed)")
    parser.add_argument("--model-column-index", type=int, help="Column index for model scores")
    parser.add_argument("--benchmark", help="Extract only a specific benchmark")
    parser.add_argument("--source-user", help="HF username/org for attribution")
    parser.add_argument("--filename", default="model_card.yaml", help="Output filename")
    parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry run)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print YAML output")
    args = parser.parse_args()

    if args.inspect:
        inspect_tables(args.repo_id)
        return

    metrics = extract_from_readme(
        repo_id=args.repo_id,
        table_index=args.table,
        model_column_index=args.model_column_index,
        benchmark_name=args.benchmark,
    )

    if not metrics:
        sys.exit(1)

    source_url = f"https://huggingface.co/{args.repo_id}"
    eval_results = convert_to_eval_results_format(
        metrics=metrics,
        source_url=source_url,
        source_name="Model Card",
        source_user=args.source_user,
    )

    if not eval_results:
        print("No benchmarks could be mapped to Hub dataset IDs", file=sys.stderr)
        sys.exit(1)

    import yaml
    indent = 2 if args.pretty else None
    print("\nExtracted evaluations (.eval_results/ format):")
    print(yaml.dump(eval_results, sort_keys=False, allow_unicode=True, default_flow_style=False))

    if args.apply or args.create_pr:
        upload_eval_results(
            repo_id=args.repo_id,
            results=eval_results,
            filename=args.filename,
            create_pr=args.create_pr,
            commit_message="Extract evaluation results from model card"
        )


if __name__ == "__main__":
    main()
