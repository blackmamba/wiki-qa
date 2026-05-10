#!/usr/bin/env python3
"""Compare two eval result files side-by-side.

Usage:
    python eval/compare.py results/run_A.json results/run_B.json
    python eval/compare.py results/run_A.json results/run_B.json --model claude-sonnet-4-6
    python eval/compare.py results/run_A.json results/run_B.json --category time_sensitive
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import citation_rate, safe_avg, short_model_name

# Scores shown in every comparison table.  Add new dimensions here to surface
# them in aggregate and per-category views automatically.
SCORE_KEYS = [
    ("correctness_score",      "Correct"),
    ("not_found_score",        "NotFound"),
    ("temporal_hedging_score", "Temporal"),
]

# Threshold below which a score delta is displayed as "·" (no meaningful change).
DELTA_EPSILON = 0.005

# Column widths used throughout the output tables.
_HEADER_COL = 10
_SECTION_WIDTH = 78


def load(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a result JSON file and return ``(meta, full_data)``.

    Args:
        path: Path to a result file produced by ``run_eval.py``.

    Returns:
        A tuple of ``(meta_dict, full_data_dict)`` where ``meta_dict`` is the
        ``_meta`` block (label, timestamp, models) and ``full_data_dict`` is
        the entire file contents (keyed by model ID plus ``"_meta"``).
    """
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    meta = data.get("_meta", {})
    return meta, data


def pick_model_results(
    data: dict[str, Any],
    model: Optional[str],
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(model_name, case_results)`` from a result file.

    Args:
        data: Full result file contents (as returned by :func:`load`).
        model: Model ID to select, or ``None`` to use the first available.

    Returns:
        A tuple of the model name string and its list of per-case result dicts.
    """
    model_keys = [k for k in data if not k.startswith("_")]
    if not model_keys:
        sys.exit("No model results found in file.")
    if model:
        if model not in model_keys:
            sys.exit(f"Model '{model}' not in file. Available: {model_keys}")
        return model, data[model]
    return model_keys[0], data[model_keys[0]]


def avg(cases: list[dict[str, Any]], key: str) -> Optional[float]:
    """Return the mean of ``key`` across *cases*, excluding ``-1`` sentinels.

    Returns ``None`` if no cases contain *key* (or all values are -1).
    """
    vals = [c[key] for c in cases if key in c and c[key] >= 0]
    return safe_avg(vals)


def fmt(val: Optional[float]) -> str:
    """Format a score value for table display."""
    return f"{val:.2f}" if val is not None else " — "


def delta(a: Optional[float], b: Optional[float]) -> str:
    """Format the signed delta B−A, or '·' when the change is negligible."""
    if a is None or b is None:
        return ""
    d = b - a
    if abs(d) < DELTA_EPSILON:
        return "  ·"
    return f" {'+' if d > 0 else ''}{d:.2f}"


def print_comparison(
    path_a: str,
    path_b: str,
    model: Optional[str],
    category_filter: Optional[str],
) -> None:
    """Print a full side-by-side comparison of two eval result files.

    Sections printed:
      1. Header (run labels, timestamps, case counts)
      2. Aggregate metric comparison with Δ column
      3. Per-category score breakdown
      4. Per-case regressions (▼) and improvements (▲)
      5. Cases present only in B

    Args:
        path_a: Path to the baseline result file.
        path_b: Path to the new result file.
        model: Model ID to compare; defaults to the first model in each file.
        category_filter: If set, restrict comparison to this category only.
    """
    meta_a, data_a = load(path_a)
    meta_b, data_b = load(path_b)

    model_a, results_a = pick_model_results(data_a, model)
    model_b, results_b = pick_model_results(data_b, model)

    label_a = meta_a.get("label", Path(path_a).stem)
    label_b = meta_b.get("label", Path(path_b).stem)

    if category_filter:
        results_a = [r for r in results_a if r["category"] == category_filter]
        results_b = [r for r in results_b if r["category"] == category_filter]
        if not results_a:
            sys.exit(f"No cases with category '{category_filter}' in {path_a}")

    by_id_a = {r["id"]: r for r in results_a}
    by_id_b = {r["id"]: r for r in results_b}
    common_ids = sorted(set(by_id_a) & set(by_id_b))
    only_in_b = sorted(set(by_id_b) - set(by_id_a))

    col = _HEADER_COL
    header_a = f"{label_a}/{short_model_name(model_a)}"[:col]
    header_b = f"{label_b}/{short_model_name(model_b)}"[:col]

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{'=' * _SECTION_WIDTH}")
    print(f"  A: {label_a}  ({meta_a.get('timestamp', '?')})  model={model_a}")
    print(f"  B: {label_b}  ({meta_b.get('timestamp', '?')})  model={model_b}")
    if category_filter:
        print(f"  Filter: category={category_filter}")
    print(f"  Common cases: {len(common_ids)}  |  New in B: {len(only_in_b)}")
    print(f"{'=' * _SECTION_WIDTH}")

    # ── Aggregate comparison ───────────────────────────────────────────────
    print(f"\n  {'Metric':<28} {'A':>{col}} {'B':>{col}} {'Δ (B−A)':>10}")
    print(f"  {'─' * 58}")

    a_cases = [by_id_a[i] for i in common_ids]
    b_cases = [by_id_b[i] for i in common_ids]
    for key, label in SCORE_KEYS:
        a_val = avg(a_cases, key)
        b_val = avg(b_cases, key)
        if a_val is not None or b_val is not None:
            print(f"  {label + ' (avg/2)':<28} {fmt(a_val):>{col}} {fmt(b_val):>{col}} {delta(a_val, b_val):>10}")

    # Search rate
    a_rate = sum(1 for i in common_ids if by_id_a[i].get("searched")) / len(common_ids)
    b_rate = sum(1 for i in common_ids if by_id_b[i].get("searched")) / len(common_ids)
    print(f"  {'Search rate':<28} {a_rate:>{col}.0%} {b_rate:>{col}.0%} {delta(a_rate, b_rate):>10}")

    # Citation rate (expected cases only)
    a_cite = citation_rate(a_cases)
    b_cite = citation_rate(b_cases)
    print(f"  {'Citation rate (expected)':<28} {a_cite:>{col}.0%} {b_cite:>{col}.0%} {delta(a_cite, b_cite):>10}")

    a_searches = sum(by_id_a[i]["search_count"] for i in common_ids)
    b_searches = sum(by_id_b[i]["search_count"] for i in common_ids)
    print(f"  {'Total searches':<28} {a_searches:>{col}} {b_searches:>{col}} {delta(float(a_searches), float(b_searches)):>10}")

    # ── Per-category breakdown ─────────────────────────────────────────────
    categories = sorted({by_id_a[i]["category"] for i in common_ids})
    print(f"\n  {'Category':<20} {'Metric':<16} {'A':>{col}} {'B':>{col}} {'Δ':>8}")
    print(f"  {'─' * 62}")
    for cat in categories:
        cat_a = [by_id_a[i] for i in common_ids if by_id_a[i]["category"] == cat]
        cat_b = [by_id_b[i] for i in common_ids if by_id_b[i]["category"] == cat]
        first = True
        for key, label in SCORE_KEYS:
            a_val = avg(cat_a, key)
            b_val = avg(cat_b, key)
            if a_val is not None or b_val is not None:
                cat_str = cat if first else ""
                first = False
                print(f"  {cat_str:<20} {label:<16} {fmt(a_val):>{col}} {fmt(b_val):>{col}} {delta(a_val, b_val):>8}")
        if first:
            print(f"  {cat:<20} {'(no scores)':<16}")

    # ── Per-case regressions / improvements ───────────────────────────────
    changes = []
    for i in common_ids:
        ra, rb = by_id_a[i], by_id_b[i]
        for key, label in SCORE_KEYS:
            if key in ra and key in rb and ra[key] != rb[key]:
                changes.append((i, ra["category"], ra["question"][:55], label, ra[key], rb[key]))

    if changes:
        print(f"\n  Case-level changes (score differs between A and B):")
        print(f"  {'─' * 70}")
        for cid, cat, q, metric, sa, sb in changes:
            arrow = "▲" if sb > sa else "▼"
            print(f"  [{cid:02d}] {cat:<18} {metric:<10} {sa}→{sb} {arrow}  \"{q}\"")
    else:
        print("\n  No per-case score changes between A and B.")

    # ── New cases (only in B) ─────────────────────────────────────────────
    if only_in_b:
        print(f"\n  New cases in B (not in A):")
        print(f"  {'─' * 70}")
        for i in only_in_b:
            r = by_id_b[i]
            scores = [f"{label}={r[key]}/2" for key, label in SCORE_KEYS if key in r]
            print(f"  [{i:02d}] {r['category']:<18} {' '.join(scores) or '(no scores yet)'}  \"{r['question'][:45]}\"")


def main() -> None:
    """Parse CLI arguments and invoke :func:`print_comparison`."""
    parser = argparse.ArgumentParser(description="Compare two eval result files")
    parser.add_argument("file_a", help="Baseline result file (JSON)")
    parser.add_argument("file_b", help="New result file to compare against baseline")
    parser.add_argument("--model", help="Which model to compare (default: first in file)")
    parser.add_argument("--category", help="Filter to a specific category")
    args = parser.parse_args()
    print_comparison(args.file_a, args.file_b, args.model, args.category)


if __name__ == "__main__":
    main()
