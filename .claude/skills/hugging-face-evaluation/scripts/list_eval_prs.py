# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub>=1.1.4",
# ]
# ///
"""
List open PRs on Hugging Face that contain evaluation results.

Returns JSON array of PRs with:
  - user: PR author username
  - date: PR creation date (ISO format)
  - model_id: Model repository ID
  - dataset_id: Benchmark dataset ID (from .eval_results/*.yaml)
  - eval_yaml_url: URL to the eval YAML file in the PR

Usage:
    uv run scripts/list_eval_prs.py --user nielsr
    uv run scripts/list_eval_prs.py --model "meta-llama/*"
    uv run scripts/list_eval_prs.py --limit 50
"""

import argparse
import fnmatch
import json
import sys
from datetime import datetime
from typing import Any

from huggingface_hub import HfApi


def get_eval_prs(
    user: str | None = None,
    model_pattern: str | None = None,
    limit: int = 100,
    include_merged: bool = False,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Get all open PRs that contain .eval_results/ files.

    Args:
        user: Filter PRs by author username
        model_pattern: Optional glob pattern to filter models
        limit: Maximum number of models to scan
        include_merged: Include merged PRs (default: only open)
        verbose: Print progress to stderr

    Returns:
        List of PR info dicts
    """
    api = HfApi()
    results = []

    if verbose:
        print(f"Fetching top {limit} trending models...", file=sys.stderr)

    models = list(api.list_models(
        pipeline_tag="text-generation",
        sort="trendingScore",
        limit=limit,
    ))

    if model_pattern:
        models = [m for m in models if fnmatch.fnmatch(m.id, model_pattern)]

    if verbose:
        print(f"Scanning {len(models)} models for eval PRs...", file=sys.stderr)

    for i, model in enumerate(models):
        if verbose:
            print(f"  [{i+1}/{len(models)}] {model.id}", file=sys.stderr)

        try:
            prs = list(api.get_repo_discussions(
                repo_id=model.id,
                repo_type="model",
                discussion_type="pull_request",
                discussion_status="all" if include_merged else "open",
            ))
        except Exception as e:
            if verbose:
                print(f"    Skipped: {e}", file=sys.stderr)
            continue

        for pr in prs:
            if not pr.is_pull_request:
                continue
            if pr.status == "closed":
                continue
            if pr.status == "merged" and not include_merged:
                continue
            if user and pr.author != user:
                continue

            # Check for .eval_results/ files using list_repo_tree
            eval_files = get_eval_files_in_pr(api, model.id, pr.num)

            if not eval_files:
                continue

            if verbose:
                print(f"    Found PR #{pr.num}: {pr.title}", file=sys.stderr)

            created = pr.created_at
            date_str = created.isoformat() if isinstance(created, datetime) else str(created)

            for eval_file in eval_files:
                results.append({
                    "user": pr.author,
                    "date": date_str,
                    "model_id": model.id,
                    "pr_num": pr.num,
                    "pr_title": pr.title,
                    "pr_status": pr.status,
                    "dataset_id": eval_file.get("dataset_id"),
                    "eval_yaml_url": eval_file.get("url"),
                })

    return results


def get_eval_files_in_pr(
    api: HfApi,
    repo_id: str,
    pr_num: int,
) -> list[dict[str, str]]:
    """
    Get .eval_results/*.yaml files in a PR.

    Args:
        api: HfApi instance
        repo_id: Model repository ID
        pr_num: PR number

    Returns:
        List of dicts with url and dataset_id (if extractable)
    """
    results = []

    try:
        files = list(api.list_repo_tree(
            repo_id=repo_id,
            revision=f"refs/pr/{pr_num}",
            repo_type="model",
            recursive=True,
        ))
    except Exception:
        return results

    for f in files:
        if not f.path.startswith(".eval_results/"):
            continue
        if not f.path.endswith((".yaml", ".yml")):
            continue
        # Skip folders
        if hasattr(f, "tree_id"):
            continue

        url = f"https://huggingface.co/{repo_id}/blob/refs%2Fpr%2F{pr_num}/{f.path}"

        # Try to extract dataset_id from filename (e.g., gpqa.yaml -> gpqa)
        filename = f.path.split("/")[-1].replace(".yaml", "").replace(".yml", "")

        results.append({
            "url": url,
            "dataset_id": filename,
        })

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="List open PRs on Hugging Face with evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run scripts/list_eval_prs.py --user nielsr
    uv run scripts/list_eval_prs.py --model "meta-llama/*"
    uv run scripts/list_eval_prs.py --limit 20 --verbose
        """,
    )
    parser.add_argument(
        "--user", "-u",
        help="Filter PRs by author username",
    )
    parser.add_argument(
        "--model", "-m",
        help="Glob pattern to filter models (e.g., 'meta-llama/*')",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Max models to scan (default: 100)",
    )
    parser.add_argument(
        "--include-merged",
        action="store_true",
        help="Include merged PRs (default: only open)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress to stderr",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    prs = get_eval_prs(
        user=args.user,
        model_pattern=args.model,
        limit=args.limit,
        include_merged=args.include_merged,
        verbose=args.verbose,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(prs, indent=indent))


if __name__ == "__main__":
    main()
