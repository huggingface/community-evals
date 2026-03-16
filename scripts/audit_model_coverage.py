# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "python-dotenv>=1.2.1",
#     "requests>=2.32.5",
#     "tabulate>=0.9.0",
# ]
# ///

"""
Audit community evaluation coverage for trending text-generation models.

The script fetches trending models from the Hugging Face Hub, inspects each
model repository for `.eval_results/*.yaml`, and reports which supported
benchmarks have community eval scores recorded.

Examples:
    uv run scripts/audit_model_coverage.py
    uv run scripts/audit_model_coverage.py --limit 50 --min-likes 250
    uv run scripts/audit_model_coverage.py --output json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from tabulate import tabulate

REQUEST_TIMEOUT = 30
DEFAULT_LIMIT = 30
HF_MODEL_BASE_URL = "https://huggingface.co"
CHECKMARK = "✅"
CROSSMARK = "❌"


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    dataset_id: str


@dataclass
class ModelCoverage:
    repo_id: str
    likes: int
    downloads: int | None
    trending_score: int | None
    yaml_files: list[str]
    covered_dataset_ids: set[str]
    error: str | None = None

    @property
    def missing_dataset_ids(self) -> list[str]:
        return [spec.dataset_id for spec in SUPPORTED_BENCHMARKS if spec.dataset_id not in self.covered_dataset_ids]

    @property
    def missing_labels(self) -> list[str]:
        return [DATASET_LABELS[dataset_id] for dataset_id in self.missing_dataset_ids]

    @property
    def missing_count(self) -> int:
        return len(self.missing_dataset_ids)


SUPPORTED_BENCHMARKS = [
    BenchmarkSpec(label="HLE", dataset_id="cais/hle"),
    BenchmarkSpec(label="GPQA", dataset_id="Idavidrein/gpqa"),
    BenchmarkSpec(label="MMLU-Pro", dataset_id="TIGER-Lab/MMLU-Pro"),
    BenchmarkSpec(label="GSM8K", dataset_id="openai/gsm8k"),
    BenchmarkSpec(label="olmOCR-bench", dataset_id="allenai/olmOCR-bench"),
    BenchmarkSpec(label="SWE-bench Verified", dataset_id="SWE-bench/SWE-bench_Verified"),
    BenchmarkSpec(label="terminal-bench 2.0", dataset_id="terminal-bench/terminal-bench-2.0"),
]

SUPPORTED_DATASET_IDS = {spec.dataset_id for spec in SUPPORTED_BENCHMARKS}
DATASET_LABELS = {spec.dataset_id: spec.label for spec in SUPPORTED_BENCHMARKS}
INLINE_DATASET_ID_PATTERN = re.compile(
    r"dataset:\s*\{[^}]*\bid:\s*['\"]?([^,'\"}\s#]+)['\"]?",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit .eval_results coverage for trending text-generation models on the Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of trending models to fetch from the Hub API before filtering.",
    )
    parser.add_argument(
        "--min-likes",
        type=int,
        default=0,
        help="Filter out models with fewer than this many likes.",
    )
    parser.add_argument(
        "--output",
        choices=("table", "markdown", "json"),
        default="table",
        help="Output format.",
    )
    return parser.parse_args()


def build_session(hf_token: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "community-evals/audit-model-coverage"})
    if hf_token:
        session.headers.update({"Authorization": f"Bearer {hf_token}"})
    return session


def fetch_trending_models(api: HfApi, limit: int, hf_token: str | None) -> list[Any]:
    return list(
        api.list_models(
            pipeline_tag="text-generation",
            sort="trending",
            limit=limit,
            expand=["likes", "downloads", "trendingScore"],
            token=hf_token,
        )
    )


def list_eval_yaml_files(api: HfApi, repo_id: str, hf_token: str | None) -> list[str]:
    try:
        tree = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=".eval_results",
            recursive=True,
            repo_type="model",
            token=hf_token,
        )
        return sorted(item.path for item in tree if item.path.endswith((".yaml", ".yml")))
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return []
        raise


def fetch_text_file(session: requests.Session, repo_id: str, path_in_repo: str) -> str:
    repo_path = quote(repo_id, safe="/")
    file_path = quote(path_in_repo, safe="/")
    url = f"{HF_MODEL_BASE_URL}/{repo_path}/resolve/main/{file_path}"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def split_yaml_entries(content: str) -> list[list[str]]:
    entries: list[list[str]] = []
    current: list[str] = []

    for line in content.splitlines():
        if re.match(r"^\s*-\s", line):
            if current:
                entries.append(current)
            current = [line]
            continue
        if current:
            current.append(line)

    if current:
        entries.append(current)

    if entries:
        return entries

    non_empty_lines = [line for line in content.splitlines() if line.strip()]
    return [non_empty_lines] if non_empty_lines else []


def clean_yaml_scalar(raw_value: str) -> str:
    value = raw_value.strip()
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    value = value.rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def extract_dataset_id_from_entry(lines: list[str]) -> str | None:
    joined = "\n".join(lines)
    inline_match = INLINE_DATASET_ID_PATTERN.search(joined)
    if inline_match:
        dataset_id = clean_yaml_scalar(inline_match.group(1))
        if dataset_id in SUPPORTED_DATASET_IDS:
            return dataset_id

    in_dataset_block = False
    dataset_indent = -1

    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        normalized = stripped[2:].strip() if stripped.startswith("- ") else stripped

        if in_dataset_block and indent <= dataset_indent:
            in_dataset_block = False

        if normalized == "dataset:":
            in_dataset_block = True
            dataset_indent = indent
            continue

        if in_dataset_block and normalized.startswith("id:"):
            dataset_id = clean_yaml_scalar(normalized.split(":", 1)[1])
            if dataset_id in SUPPORTED_DATASET_IDS:
                return dataset_id

    return None


def entry_has_value(lines: list[str]) -> bool:
    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        stripped = raw_line.strip()
        normalized = stripped[2:].strip() if stripped.startswith("- ") else stripped
        if normalized.startswith("value:"):
            return clean_yaml_scalar(normalized.split(":", 1)[1]) != ""

    return False


def extract_dataset_ids_from_yaml(content: str) -> set[str]:
    dataset_ids: set[str] = set()

    for entry_lines in split_yaml_entries(content):
        dataset_id = extract_dataset_id_from_entry(entry_lines)
        if dataset_id and entry_has_value(entry_lines):
            dataset_ids.add(dataset_id)

    return dataset_ids


def audit_model(api: HfApi, session: requests.Session, model: Any, hf_token: str | None) -> ModelCoverage:
    repo_id = model.id
    likes = int(getattr(model, "likes", 0) or 0)
    downloads = getattr(model, "downloads", None)
    trending_score = getattr(model, "trending_score", None)

    try:
        yaml_files = list_eval_yaml_files(api, repo_id, hf_token)
    except HfHubHTTPError as exc:
        return ModelCoverage(
            repo_id=repo_id,
            likes=likes,
            downloads=downloads,
            trending_score=trending_score,
            yaml_files=[],
            covered_dataset_ids=set(),
            error=f"Could not inspect repository tree: {exc}",
        )

    if not yaml_files:
        return ModelCoverage(
            repo_id=repo_id,
            likes=likes,
            downloads=downloads,
            trending_score=trending_score,
            yaml_files=[],
            covered_dataset_ids=set(),
        )

    covered_dataset_ids: set[str] = set()
    fetch_errors: list[str] = []

    for yaml_file in yaml_files:
        try:
            content = fetch_text_file(session, repo_id, yaml_file)
        except requests.RequestException as exc:
            fetch_errors.append(f"{yaml_file}: {exc}")
            continue
        covered_dataset_ids.update(extract_dataset_ids_from_yaml(content))

    return ModelCoverage(
        repo_id=repo_id,
        likes=likes,
        downloads=downloads,
        trending_score=trending_score,
        yaml_files=yaml_files,
        covered_dataset_ids=covered_dataset_ids,
        error="; ".join(fetch_errors) if fetch_errors else None,
    )


def build_coverage_rows(results: list[ModelCoverage]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for result in results:
        row: list[Any] = [result.repo_id, result.likes, len(result.yaml_files)]
        for benchmark in SUPPORTED_BENCHMARKS:
            row.append(CHECKMARK if benchmark.dataset_id in result.covered_dataset_ids else CROSSMARK)
        rows.append(row)
    return rows


def build_gap_rows(results: list[ModelCoverage]) -> list[list[Any]]:
    gap_rows: list[list[Any]] = []
    for result in sorted(results, key=lambda item: (-item.missing_count, -item.likes, item.repo_id.lower())):
        if result.missing_count == 0:
            continue
        gap_rows.append(
            [
                result.repo_id,
                result.likes,
                result.missing_count,
                ", ".join(result.missing_labels),
            ]
        )
    return gap_rows


def build_json_payload(args: argparse.Namespace, results: list[ModelCoverage], fetched_count: int) -> dict[str, Any]:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_tag": "text-generation",
        "sort": "trending",
        "limit": args.limit,
        "min_likes": args.min_likes,
        "fetched_models": fetched_count,
        "included_models": len(results),
        "models_with_eval_results_yaml": sum(1 for result in results if result.yaml_files),
        "fully_covered_models": sum(1 for result in results if result.missing_count == 0),
        "models_with_errors": sum(1 for result in results if result.error),
    }

    models = []
    for result in results:
        coverage = {
            benchmark.label: benchmark.dataset_id in result.covered_dataset_ids for benchmark in SUPPORTED_BENCHMARKS
        }
        models.append(
            {
                "repo_id": result.repo_id,
                "likes": result.likes,
                "downloads": result.downloads,
                "trending_score": result.trending_score,
                "yaml_files": result.yaml_files,
                "covered_dataset_ids": sorted(result.covered_dataset_ids),
                "coverage": coverage,
                "missing_benchmarks": result.missing_labels,
                "missing_count": result.missing_count,
                "error": result.error,
            }
        )

    return {
        "summary": summary,
        "benchmarks": [benchmark.__dict__ for benchmark in SUPPORTED_BENCHMARKS],
        "models": models,
        "gaps": [
            {
                "repo_id": result.repo_id,
                "likes": result.likes,
                "missing_count": result.missing_count,
                "missing_benchmarks": result.missing_labels,
            }
            for result in sorted(results, key=lambda item: (-item.missing_count, -item.likes, item.repo_id.lower()))
            if result.missing_count > 0
        ],
    }


def render_table(args: argparse.Namespace, results: list[ModelCoverage], fetched_count: int, tablefmt: str) -> str:
    coverage_headers = ["Model", "Likes", "YAMLs", *[benchmark.label for benchmark in SUPPORTED_BENCHMARKS]]
    coverage_rows = build_coverage_rows(results)

    lines = [
        f"Coverage audit for trending text-generation models",
        f"Fetched {fetched_count} models, kept {len(results)} after --min-likes {args.min_likes}.",
        "",
        "Coverage matrix",
        tabulate(coverage_rows, headers=coverage_headers, tablefmt=tablefmt, disable_numparse=True),
        "",
        "Gaps summary",
    ]

    gap_rows = build_gap_rows(results)
    if gap_rows:
        lines.append(
            tabulate(
                gap_rows,
                headers=["Model", "Likes", "Missing", "Missing Benchmarks"],
                tablefmt=tablefmt,
                disable_numparse=True,
            )
        )
    else:
        lines.append("All included models have results for every supported benchmark.")

    error_rows = [[result.repo_id, result.error] for result in results if result.error]
    if error_rows:
        lines.extend(
            [
                "",
                "Fetch errors",
                tabulate(error_rows, headers=["Model", "Error"], tablefmt=tablefmt, disable_numparse=True),
            ]
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN is not set; proceeding unauthenticated.", file=sys.stderr)

    api = HfApi()
    session = build_session(hf_token)

    fetched_models = fetch_trending_models(api, args.limit, hf_token)
    models = [model for model in fetched_models if int(getattr(model, "likes", 0) or 0) >= args.min_likes]
    results = [audit_model(api, session, model, hf_token) for model in models]

    if args.output == "json":
        print(json.dumps(build_json_payload(args, results, len(fetched_models)), indent=2, ensure_ascii=False))
    elif args.output == "markdown":
        print(render_table(args, results, len(fetched_models), tablefmt="github"))
    else:
        print(render_table(args, results, len(fetched_models), tablefmt="simple"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
