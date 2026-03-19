# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31.0",
# ]
# ///

"""
Check for open pull requests on a Hugging Face model repository.

Usage:
    uv run scripts/check_prs.py --repo-id "org/model-name"
"""

import argparse
import sys
from typing import Any

import requests


def get_open_prs(repo_id: str) -> list[dict[str, Any]]:
    """
    Fetch open pull requests for a Hugging Face model repository.

    Args:
        repo_id: Hugging Face model repository ID (e.g., "nvidia/model-name")

    Returns:
        List of open PR dictionaries with num, title, author, and createdAt
    """
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
        print(f"Error fetching PRs from Hugging Face: {e}", file=sys.stderr)
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


def main():
    parser = argparse.ArgumentParser(
        description="Check for open pull requests on a Hugging Face model repository.",
        epilog="Always run this before creating new PRs to avoid duplicates.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF repository ID (e.g., 'nvidia/model-name')",
    )

    args = parser.parse_args()
    list_open_prs(args.repo_id)


if __name__ == "__main__":
    main()
