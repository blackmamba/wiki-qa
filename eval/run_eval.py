#!/usr/bin/env python3
"""Run the Wikipedia QA eval suite and print a summary."""

import argparse
import json
import os
import sys
from datetime import datetime

import anthropic
from dotenv import dotenv_values
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import score_calibration, score_correctness
from wiki_qa import run_query

os.environ.update(dotenv_values(Path(__file__).parent.parent / ".env"))

CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def has_citation(answer: str) -> bool:
    return "**Source" in answer or "wikipedia.org/wiki/" in answer


def run_eval(models: list[str]) -> None:
    with open(CASES_PATH) as f:
        cases = json.load(f)

    client = anthropic.Anthropic()
    all_results = {}

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model}  ({len(cases)} cases)")
        print("=" * 60)
        model_results = []

        for case in cases:
            cid = case["id"]
            q = case["question"]
            print(f"  [{cid:02d}] {q[:65]}...' " if len(q) > 65 else f"  [{cid:02d}] {q}")

            result = run_query(q, model=model)
            answer = result["answer"]

            row = {
                "id": cid,
                "question": q,
                "category": case["category"],
                "answer": answer,
                "searched": result["searched"],
                "search_count": result["search_count"],
                "tool_calls": result["tool_calls"],
                "citation_present": has_citation(answer),
                "model": model,
            }

            dims = case.get("eval_dimensions", [])
            if "correctness" in dims:
                j = score_correctness(q, case["reference_answer"], answer, client)
                row["correctness_score"] = j["score"]
                row["correctness_reasoning"] = j["reasoning"]

            if "calibration" in dims:
                j = score_calibration(q, answer, client)
                row["calibration_score"] = j["score"]
                row["calibration_reasoning"] = j["reasoning"]

            search_ok = result["searched"] == case["should_search"]
            row["search_behavior_ok"] = search_ok

            flags = []
            if "correctness_score" in row:
                flags.append(f"correct={row['correctness_score']}/2")
            if "calibration_score" in row:
                flags.append(f"calib={row['calibration_score']}/2")
            flags.append("cited" if row["citation_present"] else "NO-CITE")
            flags.append("searched" if row["searched"] else "no-search")
            print(f"       → {' | '.join(flags)}")

            model_results.append(row)

        all_results[model] = model_results
        _print_summary(model, model_results, cases)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if len(models) > 1:
        _print_comparison(all_results, models)


def _print_summary(model: str, results: list, cases: list) -> None:
    correctness_scores = [r["correctness_score"] for r in results if "correctness_score" in r]
    calibration_scores = [r["calibration_score"] for r in results if "calibration_score" in r]
    search_rate = sum(1 for r in results if r["searched"]) / len(results)
    citation_rate = sum(1 for r in results if r["citation_present"]) / len(results)
    search_behavior_ok = sum(1 for r in results if r["search_behavior_ok"]) / len(results)
    avg_searches = sum(r["search_count"] for r in results) / len(results)

    print(f"\n  Summary — {model}")
    print(f"  Correctness (avg):     {sum(correctness_scores)/len(correctness_scores):.2f}/2.00  (n={len(correctness_scores)})")
    if calibration_scores:
        print(f"  Calibration (avg):     {sum(calibration_scores)/len(calibration_scores):.2f}/2.00  (n={len(calibration_scores)})")
    print(f"  Search rate:           {search_rate:.0%}")
    print(f"  Search behavior match: {search_behavior_ok:.0%}  (searched when expected and vice versa)")
    print(f"  Citation rate:         {citation_rate:.0%}")
    print(f"  Avg searches/question: {avg_searches:.1f}")

    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = []
        if "correctness_score" in r:
            cats[c].append(r["correctness_score"])
    print("\n  Correctness by category:")
    for cat, scores in cats.items():
        if scores:
            print(f"    {cat:<20} {sum(scores)/len(scores):.2f}/2  (n={len(scores)})")
        else:
            print(f"    {cat:<20} n/a (calibration-only)")


def _print_comparison(all_results: dict, models: list) -> None:
    print(f"\n{'=' * 60}")
    print("Model comparison")
    print(f"{'=' * 60}")
    header = f"{'Metric':<30}" + "".join(f"{m[-20:]:>20}" for m in models)
    print(header)

    def avg(results, key):
        vals = [r[key] for r in results if key in r]
        return f"{sum(vals)/len(vals):.2f}" if vals else "—"

    metrics = [
        ("Correctness (avg/2)", "correctness_score"),
        ("Calibration (avg/2)", "calibration_score"),
    ]
    for label, key in metrics:
        row = f"{label:<30}" + "".join(
            f"{avg(all_results[m], key):>20}" for m in models
        )
        print(row)

    for label, key in [("Search rate", "searched"), ("Citation rate", "citation_present")]:
        row = f"{label:<30}" + "".join(
            f"{sum(1 for r in all_results[m] if r[key])/len(all_results[m]):>19.0%}" for m in models
        )
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Run Wikipedia QA eval suite")
    parser.add_argument(
        "--models",
        default="claude-sonnet-4-6",
        help="Comma-separated model IDs (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    run_eval(models)


if __name__ == "__main__":
    main()
