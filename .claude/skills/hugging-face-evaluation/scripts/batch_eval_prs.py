# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "markdown-it-py>=3.0.0",
#     "python-dotenv>=1.2.1",
#     "pyyaml>=6.0.3",
#     "requests>=2.32.5",
#     "pypdf>=4.0.0",
# ]
# ///
"""
Batch create evaluation PRs for trending HuggingFace models.

Usage:
    uv run scripts/batch_eval_prs.py [OPTIONS]

Options:
    --limit N          Number of models to process (default: 10)
    --sort FIELD       Sort by: downloads, likes, trending (default: trending)
    --benchmark NAME   Benchmark to add (default: HLE)
    --source SOURCE    Score source: model_card, aa (default: model_card)
    --dry-run          Preview without creating PRs
    --runs-dir DIR     Directory to store run results (default: runs/)

Examples:
    uv run scripts/batch_eval_prs.py --limit 5 --benchmark HLE --dry-run
    uv run scripts/batch_eval_prs.py --limit 20 --sort trending --benchmark GPQA
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, list_models

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SKILL_DIR.parent.parent.parent  # .claude/skills/hugging-face-evaluation -> repo root

sys.path.insert(0, str(SCRIPT_DIR))
from evaluation_manager import add_single_eval, load_env

# Source URLs for attribution
SOURCE_URLS = {
    "model_card": "https://huggingface.co/{repo_id}",
    "aa": "https://artificialanalysis.ai",
    "papers": "https://huggingface.co/papers",
}


def get_trending_models(
    limit: int = 10,
    sort: str = "trending",
    pipeline_tag: str = "text-generation"
) -> list[str]:
    """
    Get list of trending/popular model IDs from HuggingFace.

    Args:
        limit: Number of models to return
        sort: Sort method - 'trending', 'downloads', 'likes'
        pipeline_tag: Filter by pipeline tag (default: text-generation for LLMs)

    Returns:
        List of repository IDs
    """
    # Map sort options to API parameters
    sort_map = {
        "trending": "trending_score",
        "downloads": "downloads",
        "likes": "likes",
    }

    sort_key = sort_map.get(sort, "trending_score")

    # Fetch models filtered by pipeline_tag
    models = list_models(
        sort=sort_key,
        limit=limit,
        pipeline_tag=pipeline_tag,
        full=False,
    )

    return [model.id for model in models]


def generate_run_filename(benchmark: str, source: str) -> str:
    """
    Generate a unique filename for this run.

    Format: {benchmark}_{date}_{hash}.json
    Hash is based on timestamp for uniqueness.
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    # Create hash from full timestamp for uniqueness
    hash_input = f"{benchmark}_{source}_{now.isoformat()}"
    hash_short = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    return f"{benchmark.lower()}_{date_str}_{hash_short}.json"


def get_source_url(source: str, repo_id: str = "") -> str:
    """Get the URL for a given source."""
    url_template = SOURCE_URLS.get(source, "")
    return url_template.format(repo_id=repo_id)


def ensure_runs_dir(runs_dir: str) -> Path:
    """Ensure the runs directory exists and return its path."""
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    return runs_path


def load_previous_results(runs_dir: str, benchmark: str) -> set:
    """
    Load all previously processed repo_ids for a benchmark from existing run files.

    Returns a set of (repo_id, benchmark) tuples that have been processed.
    """
    processed = set()
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        return processed

    # Find all files matching this benchmark
    for run_file in runs_path.glob(f"{benchmark.lower()}_*.json"):
        try:
            with open(run_file) as f:
                data = json.load(f)
                for result in data.get("results", []):
                    repo_id = result.get("repo_id")
                    if repo_id:
                        processed.add((repo_id, benchmark))
        except (json.JSONDecodeError, IOError):
            continue

    return processed


def save_results(results: dict, results_file: Path) -> None:
    """Save results to JSON file."""
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Batch create evaluation PRs for trending HuggingFace models."
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of models to process")
    parser.add_argument("--sort", type=str, default="trending",
                        choices=["trending", "downloads", "likes"],
                        help="Sort method (default: trending)")
    parser.add_argument("--pipeline-tag", type=str, default="text-generation",
                        help="Filter models by pipeline tag (default: text-generation)")
    parser.add_argument("--benchmark", type=str, default="HLE",
                        help="Benchmark to add (default: HLE)")
    parser.add_argument("--source", type=str, default="model_card",
                        choices=["model_card", "aa", "papers"],
                        help="Score source (default: model_card)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without creating PRs")
    parser.add_argument("--runs-dir", type=str, default=str(REPO_ROOT / "runs"),
                        help="Directory to store run results (default: <repo-root>/runs/)")
    args = parser.parse_args()

    load_env()

    # Setup runs directory and generate filename
    runs_path = ensure_runs_dir(args.runs_dir)
    run_filename = generate_run_filename(args.benchmark, args.source)
    results_file = runs_path / run_filename

    print("=" * 50)
    print("Batch Evaluation PR Creator")
    print("=" * 50)
    print(f"Benchmark: {args.benchmark}")
    print(f"Source: {args.source}")
    print(f"Pipeline tag: {args.pipeline_tag}")
    print(f"Limit: {args.limit}")
    print(f"Sort: {args.sort}")
    print(f"Dry run: {args.dry_run}")
    print(f"Results file: {results_file}")
    print("=" * 50)
    print()

    # Load previously processed models for this benchmark
    previously_processed = load_previous_results(args.runs_dir, args.benchmark)
    if previously_processed:
        print(f"Found {len(previously_processed)} previously processed models for {args.benchmark}")
        print()

    # Initialize results for this run
    results = {
        "benchmark": args.benchmark,
        "source": args.source,
        "source_url": get_source_url(args.source),
        "sort": args.sort,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "created": datetime.now(timezone.utc).isoformat(),
        "results": []
    }

    # Get trending models
    print(f"Fetching top {args.limit} models (sorted by {args.sort})...")
    try:
        model_ids = get_trending_models(limit=args.limit, sort=args.sort, pipeline_tag=args.pipeline_tag)
    except Exception as e:
        print(f"Error fetching models: {e}")
        return 1

    print(f"Found {len(model_ids)} models:")
    for mid in model_ids[:5]:
        print(f"  - {mid}")
    if len(model_ids) > 5:
        print(f"  ... and {len(model_ids) - 5} more")
    print()

    # Process each model
    success_count = 0
    skip_count = 0
    not_found_count = 0

    for repo_id in model_ids:
        print("-" * 40)
        print(f"Processing: {repo_id}")

        # Check if already processed in previous runs
        if (repo_id, args.benchmark) in previously_processed:
            print("  Skipping: Already processed in previous run")
            skip_count += 1
            continue

        # Get source URL for this specific model
        source_url = get_source_url(args.source, repo_id)

        # Capture stdout to check result
        import io
        from contextlib import redirect_stdout

        stdout_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture):
                success = add_single_eval(
                    repo_id=repo_id,
                    benchmark_name=args.benchmark,
                    source=args.source,
                    create_pr=not args.dry_run,
                    apply=not args.dry_run,
                )

            output = stdout_capture.getvalue()

            # Parse the output to get the value
            value = None
            if "Found:" in output:
                try:
                    value_line = [l for l in output.split("\n") if "Found:" in l][0]
                    value = float(value_line.split("=")[1].strip())
                except (IndexError, ValueError):
                    pass

            if value is not None:
                print(f"  Found: {args.benchmark} = {value}")

                if args.dry_run:
                    status = "dry_run"
                    print("  Status: Would create PR (dry run)")
                elif "Pull request created" in output or "uploaded successfully" in output:
                    status = "pr_created"
                    print("  Status: PR created")
                else:
                    status = "uploaded"
                    print("  Status: Uploaded")

                success_count += 1
            else:
                print(f"  Not found: {args.benchmark} score not available")
                status = "not_found"
                not_found_count += 1

            # Record result with source URL
            results["results"].append({
                "repo_id": repo_id,
                "benchmark": args.benchmark,
                "value": value,
                "source": args.source,
                "source_url": source_url,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            print(f"  Error: {e}")
            results["results"].append({
                "repo_id": repo_id,
                "benchmark": args.benchmark,
                "value": None,
                "source": args.source,
                "source_url": source_url,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            not_found_count += 1

    # Save results
    save_results(results, results_file)

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Processed: {success_count + not_found_count + skip_count}")
    print(f"Success: {success_count}")
    print(f"Not found: {not_found_count}")
    print(f"Skipped (previous runs): {skip_count}")
    print(f"Results saved to: {results_file}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
