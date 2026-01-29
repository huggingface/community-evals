# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "python-dotenv>=1.0.0",
#     "pyyaml>=6.0.0",
#     "requests>=2.31.0",
# ]
# ///
"""
Import evaluation results from Artificial Analysis API.

Usage:
  # Look up a specific benchmark (dry run - prints YAML)
  AA_API_KEY=... uv run scripts/import_aa.py --repo-id "org/model" --benchmark HLE

  # Look up a benchmark and create PR
  AA_API_KEY=... uv run scripts/import_aa.py --repo-id "org/model" --benchmark GPQA --create-pr

  # Import all available benchmarks
  AA_API_KEY=... uv run scripts/import_aa.py --repo-id "org/model" --all

  # Provide value manually (skip lookup)
  uv run scripts/import_aa.py --repo-id "org/model" --benchmark HLE --value 22.5 --create-pr
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

import requests


AA_INDEX_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


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


def fetch_aa_models(api_key: str) -> list[dict[str, Any]]:
    response = requests.get(
        AA_INDEX_URL,
        headers={"x-api-key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return list(data.get("data", []))


def find_model_in_aa(models: list[dict[str, Any]], repo_id: str) -> dict[str, Any] | None:
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    model_name_normalized = model_name.lower().replace("-", " ").replace("_", " ")

    for model in models:
        aa_name = model.get("name", "").lower().replace("-", " ").replace("_", " ")
        aa_slug = model.get("slug", "").lower().replace("-", " ").replace("_", " ")
        if model_name_normalized in aa_name or model_name_normalized in aa_slug:
            return model

    return None


def lookup_benchmark_from_aa(
    models: list[dict[str, Any]],
    repo_id: str,
    benchmark_name: str,
) -> float | None:
    model = find_model_in_aa(models, repo_id)
    if not model:
        return None

    evaluations = model.get("evaluations", {})
    benchmark_normalized = benchmark_name.lower().replace(" ", "_").replace("-", "_")

    for key, value in evaluations.items():
        key_normalized = key.lower().replace(" ", "_").replace("-", "_")
        if benchmark_normalized == key_normalized or benchmark_normalized in key_normalized:
            if value is not None:
                return float(value)

    return None


def get_all_benchmarks_from_aa(
    models: list[dict[str, Any]],
    repo_id: str,
) -> list[dict[str, Any]]:
    model = find_model_in_aa(models, repo_id)
    if not model:
        return []

    evaluations = model.get("evaluations", {})
    metrics = []

    for key, value in evaluations.items():
        if value is not None:
            metrics.append({
                "name": key.replace("_", " ").title(),
                "type": key,
                "value": float(value),
            })

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
        commit_message = f"Add Artificial Analysis evaluation results for {model_name}"

    pr_description = """## Evaluation Results

This PR adds structured evaluation results using the new [`.eval_results/` format](https://huggingface.co/docs/hub/eval-results).

**Source:** [Artificial Analysis](https://artificialanalysis.ai)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import evaluation results from Artificial Analysis API.",
    )
    parser.add_argument("--repo-id", required=True, help="HuggingFace repository ID")
    parser.add_argument("--benchmark", help="Specific benchmark to look up (e.g., HLE, GPQA)")
    parser.add_argument("--value", type=float, help="Manually provide the score (skips AA lookup)")
    parser.add_argument("--all", action="store_true", help="Import all available benchmarks")
    parser.add_argument("--source-user", help="HF username/org for attribution")
    parser.add_argument("--filename", default="artificial_analysis.yaml", help="Output filename")
    parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry run)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print YAML output")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    load_env()

    if args.value is not None and args.benchmark:
        metrics = [{"name": args.benchmark, "type": args.benchmark.lower(), "value": args.value}]
    else:
        api_key = os.getenv("AA_API_KEY")
        if not api_key:
            print("Error: AA_API_KEY is required to query Artificial Analysis.", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print("Fetching models from Artificial Analysis...", file=sys.stderr)

        models = fetch_aa_models(api_key)

        if args.all:
            metrics = get_all_benchmarks_from_aa(models, args.repo_id)
            if not metrics:
                print(f"No benchmarks found for {args.repo_id} in Artificial Analysis", file=sys.stderr)
                sys.exit(1)
        elif args.benchmark:
            value = lookup_benchmark_from_aa(models, args.repo_id, args.benchmark)
            if value is None:
                print(f"Could not find {args.benchmark} score for {args.repo_id} in Artificial Analysis", file=sys.stderr)
                sys.exit(1)
            print(f"Found: {args.benchmark} = {value}")
            metrics = [{"name": args.benchmark, "type": args.benchmark.lower(), "value": value}]
        else:
            print("Error: Either --benchmark or --all is required", file=sys.stderr)
            sys.exit(1)

    eval_results = convert_to_eval_results_format(
        metrics=metrics,
        source_url="https://artificialanalysis.ai",
        source_name="Artificial Analysis",
        source_user=args.source_user,
    )

    if not eval_results:
        print("No benchmarks could be mapped to Hub dataset IDs", file=sys.stderr)
        sys.exit(1)

    import yaml
    print("\nImported evaluations (.eval_results/ format):")
    print(yaml.dump(eval_results, sort_keys=False, allow_unicode=True, default_flow_style=False))

    if args.apply or args.create_pr:
        upload_eval_results(
            repo_id=args.repo_id,
            results=eval_results,
            filename=args.filename,
            create_pr=args.create_pr,
        )


if __name__ == "__main__":
    main()
