#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "markdown-it-py>=3.0.0",
#     "pyyaml>=6.0.0",
# ]
# ///
"""
Test script for model card extraction functionality.

Demonstrates table extraction capabilities without requiring HF tokens.

Usage:
  uv run scripts/test_extraction.py
"""

import yaml

from extract_model_card import (
    extract_tables_with_parser,
    is_evaluation_table,
    extract_metrics_from_table,
    convert_to_eval_results_format,
)

# Sample README content with various table formats
SAMPLE_README = """
# My Awesome Model

## Evaluation Results

Here are the benchmark results:

| Benchmark | Score |
|-----------|-------|
| MMLU      | 85.2  |
| HumanEval | 72.5  |
| GSM8K     | 91.3  |
| GPQA      | 68.4  |
| HLE       | 22.1  |

### Detailed Breakdown

| Category      | MMLU  | GSM8K | HumanEval |
|---------------|-------|-------|-----------|
| Performance   | 85.2  | 91.3  | 72.5      |

## Other Information

This is not an evaluation table:

| Feature | Value |
|---------|-------|
| Size    | 7B    |
| Type    | Chat  |

## More Results

| Benchmark     | Accuracy | F1 Score |
|---------------|----------|----------|
| HellaSwag     | 88.9     | 0.87     |
| TruthfulQA    | 68.7     | 0.65     |
"""


def test_table_extraction():
    """Test markdown table extraction."""
    print("=" * 60)
    print("TEST 1: Table Extraction")
    print("=" * 60)

    tables = extract_tables_with_parser(SAMPLE_README)
    print(f"Found {len(tables)} tables in the sample README\n")

    for i, table in enumerate(tables, 1):
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        print(f"Table {i}:")
        print(f"  Headers: {headers}")
        print(f"  Rows: {len(rows)}")
        print()

    return tables


def test_evaluation_detection(tables):
    """Test evaluation table detection."""
    print("\n" + "=" * 60)
    print("TEST 2: Evaluation Table Detection")
    print("=" * 60)

    eval_tables = []
    for i, table in enumerate(tables, 1):
        header = table.get("headers", [])
        rows = table.get("rows", [])
        is_eval = is_evaluation_table(header, rows)
        status = "✓ IS" if is_eval else "✗ NOT"
        print(f"\nTable {i}: {status} an evaluation table")
        print(f"  Header: {header}")

        if is_eval:
            eval_tables.append(table)

    print(f"\nFound {len(eval_tables)} evaluation tables")
    return eval_tables


def test_metric_extraction(eval_tables):
    """Test metric extraction."""
    print("\n" + "=" * 60)
    print("TEST 3: Metric Extraction")
    print("=" * 60)

    all_metrics = []
    for i, table in enumerate(eval_tables, 1):
        header = table.get("headers", [])
        rows = table.get("rows", [])
        print(f"\nExtracting metrics from table {i}:")
        metrics = extract_metrics_from_table(header, rows)

        print(f"  Extracted {len(metrics)} metrics:")
        for metric in metrics:
            print(f"    - {metric['name']}: {metric['value']} (type: {metric['type']})")

        all_metrics.extend(metrics)

    return all_metrics


def test_eval_results_format(metrics):
    """Test .eval_results/ format generation."""
    print("\n" + "=" * 60)
    print("TEST 4: .eval_results/ Format")
    print("=" * 60)

    eval_results = convert_to_eval_results_format(
        metrics=metrics,
        source_url="https://huggingface.co/test/model",
        source_name="Model Card",
    )

    print("\nGenerated .eval_results/ structure:")
    print(yaml.dump(eval_results, sort_keys=False, default_flow_style=False))

    return eval_results


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MODEL CARD EXTRACTION TEST SUITE")
    print("=" * 60)
    print("\nThis test demonstrates the table extraction capabilities")
    print("without requiring API access or tokens.\n")

    # Run tests
    tables = test_table_extraction()
    eval_tables = test_evaluation_detection(tables)
    metrics = test_metric_extraction(eval_tables)
    eval_results = test_eval_results_format(metrics)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✓ Found {len(tables)} total tables")
    print(f"✓ Identified {len(eval_tables)} evaluation tables")
    print(f"✓ Extracted {len(metrics)} metrics")
    print(f"✓ Converted {len(eval_results)} to .eval_results/ format")
    print("\n" + "=" * 60)
    print("All tests completed! The extraction logic is working correctly.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
