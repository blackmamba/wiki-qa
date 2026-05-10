#!/usr/bin/env python3
"""Run the Wikipedia QA eval suite and print a summary.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --models claude-sonnet-4-6,claude-haiku-4-5-20251001
    python eval/run_eval.py --cases 27,28,29,30 --label v4-honesty --verbose
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anthropic
from dotenv import dotenv_values

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import (
    JudgeResult,
    score_correctness,
    score_harmful_refusal,
    score_honesty_under_pressure,
    score_not_found,
    score_premise_correction,
    score_sycophancy,
    score_temporal_hedging,
)
from utils import citation_rate, safe_avg, short_model_name
from wiki_qa import run_query

os.environ.update(dotenv_values(Path(__file__).parent.parent / ".env"))

CASES_PATH = Path(__file__).parent / "cases.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
SECTION_WIDTH = 70
TABLE_WIDTH = 62
SUMMARY_WIDTH = 55

# ---------------------------------------------------------------------------
# Scoring dispatch table
# Each entry: dimension_key -> (scorer_fn, score_key, reasoning_key, display_label)
# Correctness is excluded here because it takes an extra ``reference`` argument;
# it is handled explicitly in the scoring loop.
# ---------------------------------------------------------------------------
_SCORERS: dict[str, tuple] = {
    "not_found":              (score_not_found,              "not_found_score",              "not_found_reasoning",              "not_found"),
    "temporal_hedging":       (score_temporal_hedging,       "temporal_hedging_score",       "temporal_hedging_reasoning",       "temporal"),
    "sycophancy":             (score_sycophancy,             "sycophancy_score",             "sycophancy_reasoning",             "sycophancy"),
    "harmful_refusal":        (score_harmful_refusal,        "harmful_refusal_score",        "harmful_refusal_reasoning",        "harmful_refusal"),
    "honesty_under_pressure": (score_honesty_under_pressure, "honesty_under_pressure_score", "honesty_under_pressure_reasoning", "honesty_pressure"),
    "premise_correction":     (score_premise_correction,     "premise_correction_score",     "premise_correction_reasoning",     "premise_correction"),
}


def has_citation(answer: str) -> bool:
    """Return True if *answer* contains a Wikipedia source attribution.

    Uses a best-effort string match: looks for the ``**Source`` markdown
    pattern emitted by the agent, or a bare ``wikipedia.org/wiki/`` URL.
    """
    return "**Source" in answer or "wikipedia.org/wiki/" in answer


def run_eval(
    models: list[str],
    verbose: bool = False,
    label: str = "",
    case_ids: Optional[list[int]] = None,
) -> None:
    """Run the full eval suite (or a filtered subset) across one or more models.

    For each model, iterates over cases, calls ``run_query``, scores each
    applicable dimension via the judge, prints a per-case summary, writes a
    timestamped JSON result file, and optionally prints a cross-model comparison.

    Args:
        models: List of Claude model IDs to evaluate.
        verbose: When True, print full answer text and judge reasoning per case.
        label: Short identifier appended to the result filename. Sanitized to
            ``[A-Za-z0-9_-]`` before use; path-traversal characters are replaced.
        case_ids: If provided, only run cases whose ``id`` is in this list.
    """
    with open(CASES_PATH) as f:
        cases = json.load(f)

    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
        if not cases:
            sys.exit(f"No cases found for IDs: {case_ids}")

    client = anthropic.Anthropic()
    all_results: dict[str, list[dict[str, Any]]] = {}

    for model in models:
        print(f"\n{'=' * SECTION_WIDTH}")
        print(f"  Model: {model}  ({len(cases)} cases)")
        print(f"{'=' * SECTION_WIDTH}")
        model_results: list[dict[str, Any]] = []

        for case in cases:
            cid = case["id"]
            q = case["question"]
            category = case["category"]

            print(f"\n  [{cid:02d}/{category}] {q}")

            result = run_query(q, model=model)
            answer = result["answer"]

            # ── Search trace ──────────────────────────────────────────────
            if result["searched"]:
                for i, tc in enumerate(result["tool_calls"], 1):
                    title = tc["result_title"] or "no result"
                    print(f"         search {i}: \"{tc['query']}\"  →  [{title}]")
            else:
                print("         search:   (skipped)")

            # ── Answer (verbose only) ─────────────────────────────────────
            if verbose:
                print("         answer:")
                for line in answer.strip().splitlines():
                    print(f"           {line}")

            citation_present = has_citation(answer)
            citation_expected = case.get("citation_expected", True)

            row: dict[str, Any] = {
                "id": cid,
                "question": q,
                "category": category,
                "answer": answer,
                "searched": result["searched"],
                "search_count": result["search_count"],
                "tool_calls": result["tool_calls"],
                "citation_present": citation_present,
                "citation_expected": citation_expected,
                "model": model,
            }

            dims = case.get("eval_dimensions", [])
            score_lines: list[tuple[str, str]] = []

            # ── Correctness (special case: needs reference_answer) ────────
            if "correctness" in dims:
                j: JudgeResult = score_correctness(
                    q, case["reference_answer"], answer, client
                )
                row["correctness_score"] = j["score"]
                row["correctness_reasoning"] = j["reasoning"]
                score_lines.append((f"correct={j['score']}/2", j["reasoning"]))

            # ── All other dimensions (data-driven dispatch) ───────────────
            for dim, (fn, score_key, reason_key, label_str) in _SCORERS.items():
                if dim in dims:
                    j = fn(q, answer, client)
                    row[score_key] = j["score"]
                    row[reason_key] = j["reasoning"]
                    score_lines.append((f"{label_str}={j['score']}/2", j["reasoning"]))

            for score_str, reasoning in score_lines:
                if verbose:
                    print(f"         {score_str}  ← {reasoning}")
                else:
                    print(f"         {score_str}")

            # ── Programmatic flags ────────────────────────────────────────
            search_ok = result["searched"] == case["should_search"]
            row["search_behavior_ok"] = search_ok

            if citation_expected:
                cited_str = "cited ✓" if citation_present else "NO-CITE ✗"
            else:
                cited_str = "cite n/a" if not citation_present else "cited (bonus)"

            search_str = f"searched {result['search_count']}x" if result["searched"] else "no-search"
            behavior_str = "" if search_ok else "  ⚠ unexpected search behavior"
            print(f"         {cited_str} | {search_str}{behavior_str}")

            model_results.append(row)

        all_results[model] = model_results
        _print_summary(model, model_results)

    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    # Sanitize label: prevent path-traversal characters in the output filename.
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label) if label else ""
    label_part = f"_{safe_label}" if safe_label else ""
    out_path = RESULTS_DIR / f"run_{timestamp}{label_part}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    output = {
        "_meta": {
            "label": label or "unlabeled",
            "timestamp": timestamp,
            "models": models,
            "n_cases": len(cases),
        },
        **all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}  (label: {label or 'unlabeled'})")

    if len(models) > 1:
        _print_comparison(all_results, models)


def _print_summary(model: str, results: list[dict[str, Any]]) -> None:
    """Print a per-model summary table: aggregate scores, rates, and per-category breakdown."""
    def _avg(key: str) -> tuple[Optional[float], int]:
        vals = [r[key] for r in results if key in r and r[key] >= 0]  # exclude -1 sentinels
        return (safe_avg(vals), len(vals))

    corr_avg, corr_n = _avg("correctness_score")
    nf_avg, nf_n = _avg("not_found_score")
    th_avg, th_n = _avg("temporal_hedging_score")
    search_rate = sum(1 for r in results if r["searched"]) / len(results)
    cite_rate = citation_rate(results)
    search_ok_rate = sum(1 for r in results if r["search_behavior_ok"]) / len(results)
    avg_searches = sum(r["search_count"] for r in results) / len(results)

    print(f"\n  {'─' * SUMMARY_WIDTH}")
    print(f"  Summary — {model}")
    print(f"  {'─' * SUMMARY_WIDTH}")
    if corr_avg is not None:
        print(f"  Correctness (avg):       {corr_avg:.2f}/2.00  (n={corr_n})")
    if nf_avg is not None:
        print(f"  Not-found acknowledgment:{nf_avg:>6.2f}/2.00  (n={nf_n})")
    if th_avg is not None:
        print(f"  Temporal hedging (avg):  {th_avg:>6.2f}/2.00  (n={th_n})")
    print(f"  Search rate:             {search_rate:.0%}  ({sum(1 for r in results if r['searched'])}/{len(results)} cases)")
    print(f"  Search behavior match:   {search_ok_rate:.0%}")
    print(f"  Citation rate (expected):{cite_rate:>5.0%}")
    print(f"  Avg searches/question:   {avg_searches:.1f}")
    print(f"  Total searches:          {sum(r['search_count'] for r in results)}")

    cats: dict[str, dict] = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"correct": [], "not_found": [], "temporal": [], "searches": 0}
        if "correctness_score" in r and r["correctness_score"] >= 0:
            cats[c]["correct"].append(r["correctness_score"])
        if "not_found_score" in r and r["not_found_score"] >= 0:
            cats[c]["not_found"].append(r["not_found_score"])
        if "temporal_hedging_score" in r and r["temporal_hedging_score"] >= 0:
            cats[c]["temporal"].append(r["temporal_hedging_score"])
        cats[c]["searches"] += r["search_count"]

    print(f"\n  {'Category':<22} {'Correct':>8} {'NotFound':>9} {'Temporal':>9} {'Searches':>9}")
    print(f"  {'─' * TABLE_WIDTH}")
    for cat, data in cats.items():
        c_str = f"{safe_avg(data['correct']):.2f}" if data["correct"] else "—"
        nf_str = f"{safe_avg(data['not_found']):.2f}" if data["not_found"] else "—"
        th_str = f"{safe_avg(data['temporal']):.2f}" if data["temporal"] else "—"
        print(f"  {cat:<22} {c_str:>8} {nf_str:>9} {th_str:>9} {data['searches']:>9}")


def _print_comparison(all_results: dict[str, list[dict[str, Any]]], models: list[str]) -> None:
    """Print a side-by-side score comparison across multiple models."""
    col = 22
    print(f"\n{'=' * SECTION_WIDTH}")
    print("  Model comparison")
    print(f"{'=' * SECTION_WIDTH}")
    short = [short_model_name(m) for m in models]
    print(f"  {'Metric':<30}" + "".join(f"{s:>{col}}" for s in short))
    print(f"  {'─' * 60}")

    def _fmt_avg(results: list[dict], key: str) -> str:
        vals = [r[key] for r in results if key in r and r[key] >= 0]
        avg = safe_avg(vals)
        return f"{avg:.2f}" if avg is not None else "—"

    for metric_label, key in [
        ("Correctness (avg/2)", "correctness_score"),
        ("Not-found ack (avg/2)", "not_found_score"),
        ("Temporal hedging (avg/2)", "temporal_hedging_score"),
    ]:
        print(f"  {metric_label:<30}" + "".join(f"{_fmt_avg(all_results[m], key):>{col}}" for m in models))

    for metric_label, key in [("Search rate", "searched"), ("Citation rate", "citation_present")]:
        print(f"  {metric_label:<30}" + "".join(
            f"{sum(1 for r in all_results[m] if r[key]) / len(all_results[m]):>{col}.0%}"
            for m in models
        ))

    print(f"  {'Total searches':<30}" + "".join(
        f"{sum(r['search_count'] for r in all_results[m]):>{col}}" for m in models
    ))


def main() -> None:
    """Parse CLI arguments and invoke :func:`run_eval`."""
    parser = argparse.ArgumentParser(description="Run Wikipedia QA eval suite")
    parser.add_argument(
        "--models",
        default="claude-sonnet-4-6",
        help="Comma-separated model IDs (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full answer text and judge reasoning per case",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Short name for this run (e.g. 'v5-temporal'). Included in the filename.",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case IDs to run (e.g. '27,28,29,30'). Runs all if omitted.",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    case_ids = [int(x.strip()) for x in args.cases.split(",") if x.strip()] if args.cases else None
    run_eval(models, verbose=args.verbose, label=args.label, case_ids=case_ids)


if __name__ == "__main__":
    main()
