# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "python-dotenv>=1.2.1",
#     "pyyaml>=6.0.3",
#     "requests>=2.32.5",
# ]
# ///

"""
Add Artificial Analysis evaluations to a Hugging Face model repository.

This script outputs evaluation results in the new .eval_results/ format
as documented at https://huggingface.co/docs/hub/eval-results

NOTE: This is a standalone reference script. For integrated functionality
with additional features (README extraction, validation, etc.), use:
    ../scripts/evaluation_manager.py import-aa [options]

STANDALONE USAGE:
AA_API_KEY="<your-api-key>" HF_TOKEN="<your-huggingface-token>" \
python artificial_analysis_to_hub.py \
--creator-slug <artificial-analysis-creator-slug> \
--model-name <artificial-analysis-model-name> \
--repo-id <huggingface-repo-id>

INTEGRATED USAGE (Recommended):
python ../scripts/evaluation_manager.py import-aa \
--creator-slug <creator-slug> \
--model-name <model-name> \
--repo-id <repo-id> \
[--create-pr]
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

import requests
import yaml
import dotenv
from huggingface_hub import HfApi

dotenv.load_dotenv()

API_KEY = os.getenv("AA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
HEADERS = {"x-api-key": API_KEY}

if not API_KEY:
    raise ValueError("AA_API_KEY is not set")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set")


def load_benchmark_mapping():
    """Load the benchmark-to-dataset mapping from metric_mapping.json."""
    mapping_file = Path(__file__).parent / "metric_mapping.json"

    if not mapping_file.exists():
        # Fallback to minimal mapping
        return {
            "MMLU": {"dataset_id": "cais/mmlu", "aliases": ["mmlu"]},
            "GPQA": {"dataset_id": "Idavidrein/gpqa", "task_id": "gpqa_diamond", "aliases": ["gpqa"]},
            "GSM8K": {"dataset_id": "openai/gsm8k", "aliases": ["gsm8k"]},
        }

    with open(mapping_file) as f:
        mapping = json.load(f)

    mapping.pop("_comment", None)
    return mapping


def find_benchmark_dataset(benchmark_name, mapping):
    """Find the Hub dataset ID for a benchmark name."""
    normalized = benchmark_name.lower().replace(" ", "_").replace("-", "_")

    # Try exact match
    if benchmark_name in mapping:
        entry = mapping[benchmark_name]
        return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    # Try case-insensitive match
    for key, entry in mapping.items():
        if key.lower() == benchmark_name.lower():
            return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    # Try matching aliases
    for key, entry in mapping.items():
        aliases = entry.get("aliases", [])
        if normalized in [a.lower().replace(" ", "_").replace("-", "_") for a in aliases]:
            return {"dataset_id": entry["dataset_id"], "task_id": entry.get("task_id", "default")}

    return None


def get_model_evaluations_data(creator_slug, model_name):
    response = requests.get(URL, headers=HEADERS)
    response_data = response.json()["data"]
    for model in response_data:
        if (
            model["model_creator"]["slug"] == creator_slug
            and model["slug"] == model_name
        ):
            return model
    raise ValueError(f"Model {model_name} not found")


def aa_evaluations_to_eval_results(model):
    """
    Convert Artificial Analysis model data to .eval_results/ format.

    Returns a list of evaluation entries ready for YAML output.
    """
    if not model:
        raise ValueError("Model data is required")

    evaluations = model.get("evaluations", {})
    mapping = load_benchmark_mapping()
    results = []
    today = date.today().isoformat()

    for key, value in evaluations.items():
        if value is None:
            continue

        # Convert key to title case for matching
        benchmark_name = key.replace("_", " ").title()
        dataset_info = find_benchmark_dataset(benchmark_name, mapping)

        if not dataset_info:
            # Try the original key as well
            dataset_info = find_benchmark_dataset(key, mapping)

        if not dataset_info:
            print(f"Warning: Could not find Hub dataset ID for '{benchmark_name}'. Skipping.")
            continue

        entry = {
            "dataset": {
                "id": dataset_info["dataset_id"],
            },
            "value": value,
            "date": today,
            "source": {
                "url": "https://artificialanalysis.ai",
                "name": "Artificial Analysis",
            }
        }

        # Add task_id if not default
        if dataset_info.get("task_id") and dataset_info["task_id"] != "default":
            entry["dataset"]["task_id"] = dataset_info["task_id"]

        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator-slug", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--filename", type=str, default="artificial_analysis.yaml",
                        help="Output filename in .eval_results/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print YAML without uploading")
    args = parser.parse_args()

    aa_evaluations_data = get_model_evaluations_data(
        creator_slug=args.creator_slug, model_name=args.model_name
    )

    eval_results = aa_evaluations_to_eval_results(model=aa_evaluations_data)

    if not eval_results:
        print("No evaluations could be mapped to Hub dataset IDs")
        return

    # Generate YAML content
    yaml_content = yaml.dump(eval_results, sort_keys=False, allow_unicode=True)

    if args.dry_run:
        print("\nGenerated .eval_results/ YAML:")
        print(yaml_content)
        return

    # Upload to .eval_results/ folder
    api = HfApi(token=HF_TOKEN)
    file_path = f".eval_results/{args.filename}"

    commit_message = f"Add Artificial Analysis evaluations for {args.model_name}"
    commit_description = (
        "This commit adds structured evaluation results in the new .eval_results/ format. "
        "Results will appear on the model page and linked benchmark leaderboards. "
        "See https://huggingface.co/docs/hub/eval-results for format details."
    )

    api.upload_file(
        path_or_fileobj=yaml_content.encode("utf-8"),
        path_in_repo=file_path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=True,
    )

    print(f"✓ Pull request created for {args.repo_id}")
    print(f"  File: {file_path}")


if __name__ == "__main__":
    main()
