# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.31.0",
# ]
# ///
"""
Fetch the top N models from the Artificial Analysis index.

Usage:
  AA_API_KEY=... uv run scripts/aa_top_models_prs.py --limit 10 --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

AA_INDEX_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


def extract_index_score(model: dict[str, Any]) -> float | None:
    for key in (
        "index",
        "overall_score",
        "aggregate_score",
        "score",
        "overall",
        "rating",
        "elo",
    ):
        value = model.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    evaluations = model.get("evaluations") or {}
    for key in ("index", "overall_score", "overall", "score"):
        value = evaluations.get(key)
        if isinstance(value, (int, float)):
            return float(value)

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


def pick_top_models(models: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    scored = []
    for model in models:
        score = extract_index_score(model)
        scored.append((score, model))

    scored.sort(key=lambda item: (item[0] is None, -(item[0] or 0)))
    return [model for _, model in scored[:limit]]


def summarize_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "aa_name": model.get("name") or model.get("display_name") or model.get("slug"),
        "aa_slug": model.get("slug"),
        "aa_creator": (model.get("model_creator") or {}).get("slug"),
        "aa_index_score": extract_index_score(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch top models from the Artificial Analysis index.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Top N models (default: 10)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    api_key = os.getenv("AA_API_KEY")
    if not api_key:
        print("AA_API_KEY is required to query Artificial Analysis.", file=sys.stderr)
        sys.exit(1)

    models = fetch_aa_models(api_key)
    top_models = pick_top_models(models, args.limit)

    if args.verbose:
        print(f"Fetched {len(top_models)} models from AA.", file=sys.stderr)

    results = [summarize_model(model) for model in top_models]
    indent = 2 if args.pretty else None
    print(json.dumps(results, indent=indent))


if __name__ == "__main__":
    main()
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "requests>=2.31.0",
# ]
# ///
"""
Plan eval-result PRs for top Artificial Analysis models.

Steps:
  1) Pull top N models from the Artificial Analysis index
  2) Find matching Hugging Face model repos
  3) Detect non-AA eval sources (model card, papers)
  4) Output a PR plan with source priority:
       model_card > papers > artificial_analysis

Usage:
  AA_API_KEY=... uv run scripts/aa_top_models_prs.py --limit 10 --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Iterable

import requests
from huggingface_hub import HfApi, ModelCard

AA_INDEX_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def tokenize(value: str) -> set[str]:
    return set(normalize_name(value).split())


def extract_index_score(model: dict[str, Any]) -> float | None:
    for key in (
        "index",
        "overall_score",
        "aggregate_score",
        "score",
        "overall",
        "rating",
        "elo",
    ):
        value = model.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    evaluations = model.get("evaluations") or {}
    for key in ("index", "overall_score", "overall", "score"):
        value = evaluations.get(key)
        if isinstance(value, (int, float)):
            return float(value)

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


def pick_top_models(models: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    scored = []
    for model in models:
        score = extract_index_score(model)
        scored.append((score, model))

    scored.sort(key=lambda item: (item[0] is None, -(item[0] or 0)))
    return [model for _, model in scored[:limit]]


def direct_repo_id(model: dict[str, Any]) -> str | None:
    for key in (
        "huggingface_id",
        "huggingface_repo",
        "huggingface_repo_id",
        "hf_repo",
        "hf_repo_id",
        "hf_model_id",
        "repo_id",
    ):
        value = model.get(key)
        if isinstance(value, str) and "/" in value:
            return value
    return None


def select_best_candidate(name: str, candidates: Iterable[str]) -> str | None:
    target_tokens = tokenize(name)
    best = None
    best_score = 0.0

    for candidate in candidates:
        repo_name = candidate.split("/")[-1]
        tokens = tokenize(repo_name)
        if not tokens:
            continue
        overlap = len(tokens & target_tokens) / max(len(tokens), 1)
        if overlap > best_score:
            best_score = overlap
            best = candidate

    if best_score >= 0.6:
        return best
    return best


def find_hf_repo(api: HfApi, model: dict[str, Any]) -> str | None:
    direct = direct_repo_id(model)
    if direct:
        return direct

    name = model.get("name") or model.get("display_name") or model.get("slug")
    if not isinstance(name, str):
        return None

    candidates = list(api.list_models(search=name, limit=5))
    if not candidates:
        return None

    candidate_ids = [c.id for c in candidates if getattr(c, "id", None)]
    return select_best_candidate(name, candidate_ids)


def extract_tables(markdown_content: str) -> list[str]:
    table_pattern = r"(\|[^\n]+\|(?:\r?\n\|[^\n]+\|)+)"
    return re.findall(table_pattern, markdown_content)


def has_eval_tables(markdown_content: str) -> bool:
    if not markdown_content:
        return False

    benchmark_keywords = [
        "mmlu",
        "gpqa",
        "hle",
        "gsm8k",
        "humaneval",
        "arc",
        "hellaswag",
        "truthfulqa",
        "winogrande",
    ]
    tables = extract_tables(markdown_content)
    for table in tables:
        lower = table.lower()
        if any(keyword in lower for keyword in benchmark_keywords) and re.search(r"\d+\.?\d*%?", table):
            return True
    return False


def find_paper_tags(info: Any) -> list[str]:
    tags = getattr(info, "tags", []) or []
    return [tag for tag in tags if tag.startswith(("arxiv:", "paper:", "papers:"))]


def plan_entry(api: HfApi, model: dict[str, Any]) -> dict[str, Any]:
    name = model.get("name") or model.get("display_name") or model.get("slug") or "unknown"
    index_score = extract_index_score(model)

    repo_id = find_hf_repo(api, model)
    if not repo_id:
        return {
            "aa_name": name,
            "aa_index_score": index_score,
            "hf_repo_id": None,
            "source_priority": None,
            "notes": "No matching Hugging Face repo found.",
            "next_steps": [],
        }

    card_text = ""
    try:
        card = ModelCard.load(repo_id)
        card_text = card.content or ""
    except Exception:
        card_text = ""

    info = None
    try:
        info = api.model_info(repo_id)
    except Exception:
        info = None

    has_tables = has_eval_tables(card_text)
    paper_tags = find_paper_tags(info) if info else []

    if has_tables:
        source_priority = "model_card"
        next_steps = [
            f'uv run scripts/evaluation_manager.py inspect-tables --repo-id "{repo_id}"',
            f'uv run scripts/evaluation_manager.py extract-readme --repo-id "{repo_id}" --table <N> --create-pr',
        ]
    elif paper_tags:
        source_priority = "papers"
        next_steps = [
            f'mcp__hf-mcp-server__hub_repo_details repo_ids: ["{repo_id}"] include_readme: true',
            "mcp__hf-mcp-server__paper_search query: \"<model name> <arxiv id>\" results_limit: 3",
            f'uv run scripts/evaluation_manager.py add-eval --benchmark <BENCH> --repo-id "{repo_id}" --value <SCORE> --create-pr',
        ]
    else:
        source_priority = "artificial_analysis"
        next_steps = [
            f'uv run scripts/evaluation_manager.py add-eval --benchmark <BENCH> --repo-id "{repo_id}" --source aa --create-pr',
        ]

    return {
        "aa_name": name,
        "aa_index_score": index_score,
        "hf_repo_id": repo_id,
        "source_priority": source_priority,
        "paper_tags": paper_tags,
        "next_steps": next_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan eval PRs for top Artificial Analysis models.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Top N models (default: 10)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    api_key = os.getenv("AA_API_KEY")
    if not api_key:
        print("AA_API_KEY is required to query Artificial Analysis.", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    models = fetch_aa_models(api_key)
    top_models = pick_top_models(models, args.limit)

    results = []
    for idx, model in enumerate(top_models, start=1):
        if args.verbose:
            name = model.get("name") or model.get("display_name") or model.get("slug") or "unknown"
            print(f"[{idx}/{len(top_models)}] {name}", file=sys.stderr)
        results.append(plan_entry(api, model))

    indent = 2 if args.pretty else None
    print(json.dumps(results, indent=indent))


if __name__ == "__main__":
    main()
