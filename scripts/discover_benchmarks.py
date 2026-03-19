#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "python-dotenv>=1.2.1",
# ]
# ///

"""
Discover candidate benchmark datasets from Hugging Face Hub.

Queries the Hub for two tiers of benchmark datasets:

  1. Official benchmarks  — Hub-registered leaderboard benchmarks
     (tag: benchmark:official). These support .eval_results/ YAML directly.
  2. Community benchmarks — datasets tagged 'benchmark' by their authors.
     Broader signal for emerging evaluations.

Cross-references both lists against benchmarks already tracked in
community-evals and outputs a ranked report of candidates for potential
addition to the workflow.

Usage:
    uv run scripts/discover_benchmarks.py
    uv run scripts/discover_benchmarks.py --min-likes 50
    uv run scripts/discover_benchmarks.py --top 50 --output-md candidates.md
    uv run scripts/discover_benchmarks.py --output-json candidates.json
    uv run scripts/discover_benchmarks.py --limit 500 --min-likes 5
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# Benchmarks already tracked in community-evals
# Sources: poll_new_evals.py (BENCHMARK_DATASETS) + metric_mapping.json
# ---------------------------------------------------------------------------

TRACKED_DATASET_IDS = {
    # Active in poll_new_evals.py
    "cais/hle",
    "Idavidrein/gpqa",
    "TIGER-Lab/MMLU-Pro",
    "openai/gsm8k",
    "allenai/olmOCR-bench",
    "SWE-bench/SWE-bench_Verified",
    "terminal-bench/terminal-bench-2.0",
    # Additional entries from metric_mapping.json
    "cais/mmlu",
    "edinburgh-dawg/mmlu-redux",
    "openai/openai_humaneval",
    "Rowan/hellaswag",
    "allenai/ai2_arc",
    "allenai/winogrande",
    "truthfulqa/truthful_qa",
    "ucinlp/drop",
    "lukaemon/bbh",
    "lighteval/MATH",
    "OpenEvals/aime_2025",
    "princeton-nlp/SWE-bench_Verified",
    "livecodebench/livecodebench",
    "OpenEvals/SimpleQA",
    "OpenEvals/aime_24",
    "google/IFEval",
    "google-research-datasets/mbpp",
    "OpenEvals/MuSR",
}

# Tags to strip from display (too noisy / not informative)
_SKIP_TAG_PREFIXES = (
    "language:",
    "size_categories:",
    "license:",
    "region:",
    "pretty_name:",
    "arxiv:",
    "doi:",
    "benchmark:",
    "format:",
    "modality:",
    "library:",
    "annotations_creators:",
    "language_creators:",
    "multilinguality:",
    "source_datasets:",
)
_SKIP_TAGS = {"benchmark", "evaluation", "datasets"}


def _info_to_entry(ds) -> dict:
    likes = getattr(ds, "likes", 0) or 0
    downloads = getattr(ds, "downloads", 0) or 0
    tags = list(getattr(ds, "tags", []) or [])
    return {
        "id": ds.id,
        "likes": likes,
        "downloads": downloads,
        "tags": tags,
        "url": f"https://huggingface.co/datasets/{ds.id}",
    }


def fetch_official(api: HfApi, limit: int) -> list:
    """Fetch Hub-registered official benchmarks (benchmark:official tag)."""
    print(f"  Fetching official benchmarks (benchmark='official')...", flush=True)
    try:
        results = list(
            api.list_datasets(
                benchmark="official",
                sort="likes",
                limit=limit,
                expand=["likes", "downloads", "tags"],
            )
        )
        print(f"  Got {len(results)} official benchmark datasets.", flush=True)
        return results
    except Exception as exc:
        print(f"  Error fetching official benchmarks: {exc}", file=sys.stderr)
        return []


def fetch_community(api: HfApi, limit: int) -> list:
    """Fetch datasets tagged 'benchmark' (community-tagged, broader signal)."""
    print(f"  Fetching up to {limit} community benchmark datasets...", flush=True)
    try:
        results = list(
            api.list_datasets(
                filter=["benchmark"],
                sort="likes",
                limit=limit,
                expand=["likes", "downloads", "tags"],
            )
        )
        print(f"  Got {len(results)} community benchmark datasets.", flush=True)
        return results
    except Exception as exc:
        print(f"  Error fetching community benchmarks: {exc}", file=sys.stderr)
        return []


def display_tags(tags: list[str]) -> str:
    """Return a short, human-readable tag string."""
    kept = []
    for t in tags:
        if t in _SKIP_TAGS:
            continue
        if any(t.startswith(p) for p in _SKIP_TAG_PREFIXES):
            continue
        kept.append(t)
        if len(kept) == 3:
            break
    return ", ".join(kept) if kept else "—"


def build_report(
    official_new: list[dict],
    community_new: list[dict],
    tracked: list[dict],
    args,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top_community = min(args.top, len(community_new))

    lines = [
        "# Benchmark Discovery Report",
        "",
        f"Generated: {now}  ",
        f"Source: HuggingFace Hub (`benchmark='official'` + `filter=['benchmark']`)",
        "",
    ]

    # --- Official benchmarks not yet tracked ---
    if official_new:
        lines += [
            "## New Official Benchmarks",
            "",
            "These are **Hub-registered leaderboard benchmarks** (`benchmark:official` tag)  ",
            "that support `.eval_results/` YAML and are not yet tracked in community-evals:",
            "",
            "| Dataset | Likes | Tags |",
            "|---------|------:|------|",
        ]
        for entry in official_new:
            tag_str = display_tags(entry["tags"])
            lines.append(
                f"| [{entry['id']}]({entry['url']}) | {entry['likes']:,} | {tag_str} |"
            )
        lines.append("")
    else:
        lines += [
            "## New Official Benchmarks",
            "",
            "_All Hub-registered official benchmarks are already tracked._",
            "",
        ]

    # --- Community candidates ---
    lines += [
        f"## Top {top_community} Community Benchmark Candidates",
        "",
        "Datasets tagged `benchmark` by their authors, sorted by likes.  ",
        f"Minimum {args.min_likes} likes. Excludes datasets already tracked.",
        "",
        "| # | Dataset | Likes | Downloads | Tags |",
        "|---|---------|------:|----------:|------|",
    ]
    for i, entry in enumerate(community_new[:top_community], 1):
        tag_str = display_tags(entry["tags"])
        lines.append(
            f"| {i} | [{entry['id']}]({entry['url']}) "
            f"| {entry['likes']:,} | {entry['downloads']:,} | {tag_str} |"
        )

    # --- Already tracked ---
    lines += [
        "",
        "## Already Tracked in community-evals",
        "",
        "| Dataset | Likes |",
        "|---------|------:|",
    ]
    for entry in sorted(tracked, key=lambda x: -x["likes"]):
        lines.append(
            f"| [{entry['id']}]({entry['url']}) | {entry['likes']:,} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Adding a New Benchmark",
        "",
        "1. Visit the candidate's Hub page and verify it supports `.eval_results/` YAML.",
        "2. Add it to `BENCHMARK_TRACKER.md` under **Prospective Benchmarks**.",
        "3. Once confirmed, add the dataset ID to `BENCHMARK_DATASETS` in `scripts/poll_new_evals.py`.",
        "4. Add a name → ID entry in `.claude/skills/community-evals/examples/metric_mapping.json`.",
        "",
        "_Run with `--output-json` to get structured data for automation._",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--limit", type=int, default=300,
        help="Max community-benchmark datasets to fetch (default: 300)",
    )
    parser.add_argument(
        "--min-likes", type=int, default=10,
        help="Minimum likes for community candidates (default: 10)",
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="How many community candidates to show (default: 30)",
    )
    parser.add_argument(
        "--output-json", metavar="FILE",
        help="Write full results as JSON to FILE",
    )
    parser.add_argument(
        "--output-md", metavar="FILE",
        help="Write markdown report to FILE",
    )
    args = parser.parse_args()

    api = HfApi(token=HF_TOKEN)
    tracked_normalized = {ds_id.lower() for ds_id in TRACKED_DATASET_IDS}

    print("Discovering benchmark datasets from Hugging Face Hub...")

    raw_official = fetch_official(api, limit=500)
    raw_community = fetch_community(api, limit=args.limit)

    # Deduplicate community set (some official ones appear there too)
    official_ids = {ds.id.lower() for ds in raw_official}

    def process(raw_datasets, *, exclude_ids: set = frozenset()) -> tuple[list, list]:
        """Split into (new_candidates, tracked_entries)."""
        new_, tracked_ = [], []
        seen = set()
        for ds in raw_datasets:
            key = ds.id.lower()
            if key in seen or key in exclude_ids:
                continue
            seen.add(key)
            entry = _info_to_entry(ds)
            if key in tracked_normalized:
                tracked_.append(entry)
            else:
                new_.append(entry)
        return new_, tracked_

    official_new, official_tracked = process(raw_official)
    community_new, community_tracked = process(
        raw_community, exclude_ids=official_ids
    )

    # Filter community candidates by min-likes
    community_new = [e for e in community_new if e["likes"] >= args.min_likes]
    community_new.sort(key=lambda x: (x["likes"], x["downloads"]), reverse=True)

    # Merge tracked lists, dedup
    all_tracked = {e["id"]: e for e in official_tracked + community_tracked}
    tracked_list = sorted(all_tracked.values(), key=lambda x: -x["likes"])

    print(f"\nSummary:")
    print(f"  Official benchmarks (benchmark:official) : {len(raw_official)}")
    print(f"    → already tracked                      : {len(official_tracked)}")
    print(f"    → new candidates                       : {len(official_new)}")
    print(f"  Community benchmarks (tag: benchmark)    : {len(raw_community)}")
    print(f"    → new candidates (≥{args.min_likes} likes): {len(community_new)}")
    print()

    report = build_report(official_new, community_new, tracked_list, args)
    print(report)

    if args.output_md:
        Path(args.output_md).write_text(report)
        print(f"\nMarkdown written to: {args.output_md}")

    if args.output_json:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "official": {
                "total": len(raw_official),
                "tracked": len(official_tracked),
                "new_candidates": official_new,
            },
            "community": {
                "total": len(raw_community),
                "new_candidates_count": len(community_new),
                "new_candidates": community_new,
            },
            "tracked": tracked_list,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"JSON written to: {args.output_json}")


if __name__ == "__main__":
    main()
