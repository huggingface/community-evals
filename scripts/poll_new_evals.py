# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "python-dotenv>=1.2.1",
#     "requests>=2.32.5",
# ]
# ///

"""
Poll HuggingFace benchmark dataset repos for new .eval_results/ submissions.

Checks community PRs on benchmark datasets for new YAML eval result files,
reports new results since last run, and optionally POSTs a summary to Slack.

Usage:
    uv run scripts/poll_new_evals.py
    uv run scripts/poll_new_evals.py --dry-run
    uv run scripts/poll_new_evals.py --since 2026-01-01
    uv run scripts/poll_new_evals.py --webhook-url https://hooks.slack.com/...
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_DATASETS = [
    "cais/hle",
    "Idavidrein/gpqa",
    "TIGER-Lab/MMLU-Pro",
    "openai/gsm8k",
    "allenai/olmOCR-bench",
    "SWE-bench/SWE-bench_Verified",
    "terminal-bench/terminal-bench-2.0",
]

SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "poll_state.json"

HF_API_BASE = "https://huggingface.co/api"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict[str, Any]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# HF Hub helpers
# ---------------------------------------------------------------------------


def get_discussions(
    dataset_id: str, hf_token: Optional[str], requests_mod: Any
) -> list[dict[str, Any]]:
    """Fetch all discussions (PRs + issues) for a dataset repo."""
    url = f"{HF_API_BASE}/datasets/{dataset_id}/discussions"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    discussions = []
    params: dict[str, Any] = {"limit": 100, "p": 0}

    while True:
        try:
            resp = requests_mod.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
        except requests_mod.RequestException as e:
            print(f"  Warning: could not fetch discussions for {dataset_id}: {e}", file=sys.stderr)
            break

        data = resp.json()
        page = data.get("discussions", [])
        discussions.extend(page)

        # HF paginates; stop when we get fewer than limit
        if len(page) < params["limit"]:
            break
        params["p"] += 1

    return discussions


def get_pr_files(
    dataset_id: str, pr_num: int, hf_token: Optional[str], requests_mod: Any
) -> list[str]:
    """Return a list of file paths changed in a PR."""
    url = f"{HF_API_BASE}/datasets/{dataset_id}/discussions/{pr_num}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    try:
        resp = requests_mod.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests_mod.RequestException as e:
        print(f"  Warning: could not fetch PR #{pr_num} details for {dataset_id}: {e}", file=sys.stderr)
        return []

    # The discussion detail endpoint includes a `changes` list with file paths
    changes = data.get("changes", [])
    return [c.get("path", "") for c in changes if c.get("path")]


def fetch_yaml_content(
    dataset_id: str, file_path: str, pr_ref: str, hf_token: Optional[str], requests_mod: Any
) -> Optional[str]:
    """Download a file from a PR branch on HF Hub."""
    # Resolve the PR ref: HF uses refs/pr/<num>
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{pr_ref}/{file_path}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    try:
        resp = requests_mod.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests_mod.RequestException:
        return None


def parse_eval_yaml(content: str) -> list[dict[str, Any]]:
    """Parse a .eval_results YAML file and return a list of result entries."""
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Core polling logic
# ---------------------------------------------------------------------------


def poll_benchmark(
    dataset_id: str,
    since_dt: datetime,
    hf_token: Optional[str],
    seen_prs: set[str],
    requests_mod: Any,
) -> list[dict[str, Any]]:
    """
    Check a benchmark dataset for new eval result PRs since `since_dt`.

    Returns a list of new result dicts:
      {benchmark, model, score, metric, date, pr_url, pr_author}
    """
    new_results = []
    short_name = dataset_id.split("/")[-1]
    print(f"  Checking {dataset_id} …")

    discussions = get_discussions(dataset_id, hf_token, requests_mod)

    for disc in discussions:
        if not disc.get("isPullRequest"):
            continue

        pr_num = disc.get("num")
        pr_key = f"{dataset_id}#{pr_num}"

        # Skip already-seen PRs
        if pr_key in seen_prs:
            continue

        # Parse creation date
        created_str = disc.get("createdAt", "")
        try:
            created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_dt = datetime.now(timezone.utc)

        if created_dt < since_dt:
            continue

        # Fetch files changed in this PR
        pr_ref = f"refs/pr/{pr_num}"
        files = get_pr_files(dataset_id, pr_num, hf_token, requests_mod)

        eval_files = [f for f in files if f.startswith(".eval_results/") and f.endswith(".yaml")]

        if not eval_files:
            # Mark as seen so we don't re-check it
            seen_prs.add(pr_key)
            continue

        pr_url = f"https://huggingface.co/datasets/{dataset_id}/discussions/{pr_num}"
        pr_author = disc.get("author", {}).get("name", "unknown")
        pr_title = disc.get("title", "")

        for eval_file in eval_files:
            content = fetch_yaml_content(dataset_id, eval_file, pr_ref, hf_token, requests_mod)
            if not content:
                continue

            entries = parse_eval_yaml(content)
            for entry in entries:
                # Each entry may have dataset + value fields
                value = entry.get("value")
                entry_date = entry.get("date", created_dt.date().isoformat())
                metric_name = entry.get("metric", {})
                if isinstance(metric_name, dict):
                    metric_name = metric_name.get("name", "score")

                # Try to infer model name from the PR title or file path
                # Common pattern: "Add eval results for <model>"
                model = pr_title
                for prefix in ("Add eval results for ", "Add evaluation results for ",
                               "Eval results for ", "Evaluation results for "):
                    if pr_title.lower().startswith(prefix.lower()):
                        model = pr_title[len(prefix):]
                        break

                new_results.append({
                    "benchmark": short_name,
                    "dataset_id": dataset_id,
                    "model": model or pr_author,
                    "score": value,
                    "metric": metric_name or "score",
                    "date": entry_date,
                    "pr_url": pr_url,
                    "pr_author": pr_author,
                    "pr_num": pr_num,
                })

        seen_prs.add(pr_key)

    return new_results


# ---------------------------------------------------------------------------
# Output / Slack
# ---------------------------------------------------------------------------


def format_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No new eval results found."

    lines = [f"New eval results ({len(results)} found):", ""]
    for r in results:
        score_str = f"{r['score']:.1f}" if isinstance(r['score'], float) else str(r['score'])
        lines.append(f"  [{r['benchmark']}] {r['model']}")
        lines.append(f"    Score: {score_str}  |  Date: {r['date']}  |  PR: {r['pr_url']}")
        lines.append("")

    return "\n".join(lines)


def post_to_slack(webhook_url: str, text: str, requests_mod: Any) -> bool:
    try:
        resp = requests_mod.post(
            webhook_url,
            json={"text": text},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except requests_mod.RequestException as e:
        print(f"Warning: Slack webhook failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll HuggingFace benchmark datasets for new .eval_results/ submissions."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be detected without updating state.",
    )
    parser.add_argument(
        "--since",
        metavar="DATE",
        help="Override look-back date (ISO format, e.g. 2026-01-01). "
             "Defaults to last-run timestamp stored in poll_state.json.",
    )
    parser.add_argument(
        "--webhook-url",
        metavar="URL",
        help="Slack incoming webhook URL to POST results summary to.",
    )
    args = parser.parse_args()

    # Load env
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        pass

    import requests  # type: ignore  # noqa: PLC0415

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set; requests will be unauthenticated.", file=sys.stderr)

    # Resolve since_dt
    state = load_state()

    if args.since:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        print(f"Using --since override: {since_dt.date().isoformat()}")
    elif "last_run" in state:
        since_dt = datetime.fromisoformat(state["last_run"])
        print(f"Checking for results since last run: {since_dt.date().isoformat()}")
    else:
        # First run: look back 7 days
        from datetime import timedelta
        since_dt = datetime.now(timezone.utc) - timedelta(days=7)
        print(f"First run — looking back 7 days (since {since_dt.date().isoformat()})")

    seen_prs: set[str] = set(state.get("seen_prs", []))
    original_seen_count = len(seen_prs)

    # Poll all benchmarks
    all_new_results: list[dict[str, Any]] = []
    print(f"\nPolling {len(BENCHMARK_DATASETS)} benchmark datasets…\n")

    for dataset_id in BENCHMARK_DATASETS:
        new = poll_benchmark(dataset_id, since_dt, hf_token, seen_prs, requests)
        all_new_results.extend(new)

    # Print summary
    print()
    summary = format_summary(all_new_results)
    print(summary)

    # Optionally POST to Slack
    if args.webhook_url and all_new_results:
        print("\nPosting to Slack…")
        ok = post_to_slack(args.webhook_url, summary, requests)
        if ok:
            print("Slack notification sent.")

    # Persist state (unless dry-run)
    if not args.dry_run:
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        state["seen_prs"] = sorted(seen_prs)
        save_state(state)
        new_seen = len(seen_prs) - original_seen_count
        print(f"\nState updated: {new_seen} new PRs recorded → {STATE_FILE}")
    else:
        print("\n[dry-run] State not updated.")


if __name__ == "__main__":
    main()
