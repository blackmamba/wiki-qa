#!/usr/bin/env python3
"""Run the Wikipedia QA eval suite and print a summary."""

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime

import anthropic
from dotenv import dotenv_values
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import (
    score_correctness,
    score_not_found,
    score_temporal_hedging,
    score_sycophancy,
    score_harmful_refusal,
    score_honesty_under_pressure,
    score_premise_correction,
)
from wiki_qa import run_query

os.environ.update(dotenv_values(Path(__file__).parent.parent / ".env"))

CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def has_citation(answer: str) -> bool:
    return "**Source" in answer or "wikipedia.org/wiki/" in answer


def _wrap(text: str, indent: int = 10, width: int = 90) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def run_eval(
    models: list[str],
    verbose: bool = False,
    label: str = "",
    case_ids: list[int] = None,
) -> None:
    with open(CASES_PATH) as f:
        cases = json.load(f)

    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
        if not cases:
            sys.exit(f"No cases found for IDs: {case_ids}")

    client = anthropic.Anthropic()
    all_results = {}

    for model in models:
        print(f"\n{'=' * 70}")
        print(f"  Model: {model}  ({len(cases)} cases)")
        print(f"{'=' * 70}")
        model_results = []

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
                print(f"         search:   (skipped)")

            # ── Answer (verbose only) ─────────────────────────────────────
            if verbose:
                print(f"         answer:")
                for line in answer.strip().splitlines():
                    print(f"           {line}")

            citation_present = has_citation(answer)
            citation_expected = case.get("citation_expected", True)

            row = {
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
            score_lines = []

            # ── Correctness judge ─────────────────────────────────────────
            if "correctness" in dims:
                j = score_correctness(q, case["reference_answer"], answer, client)
                row["correctness_score"] = j["score"]
                row["correctness_reasoning"] = j["reasoning"]
                score_lines.append((f"correct={j['score']}/2", j["reasoning"]))

            # ── Not-found acknowledgment judge ────────────────────────────
            if "not_found" in dims:
                j = score_not_found(q, answer, client)
                row["not_found_score"] = j["score"]
                row["not_found_reasoning"] = j["reasoning"]
                score_lines.append((f"not_found={j['score']}/2", j["reasoning"]))

            # ── Temporal hedging judge ────────────────────────────────────
            if "temporal_hedging" in dims:
                j = score_temporal_hedging(q, answer, client)
                row["temporal_hedging_score"] = j["score"]
                row["temporal_hedging_reasoning"] = j["reasoning"]
                score_lines.append((f"temporal={j['score']}/2", j["reasoning"]))

            # ── Sycophancy judge ──────────────────────────────────────────
            if "sycophancy" in dims:
                j = score_sycophancy(q, answer, client)
                row["sycophancy_score"] = j["score"]
                row["sycophancy_reasoning"] = j["reasoning"]
                score_lines.append((f"sycophancy={j['score']}/2", j["reasoning"]))

            # ── Harmful refusal judge ─────────────────────────────────────
            if "harmful_refusal" in dims:
                j = score_harmful_refusal(q, answer, client)
                row["harmful_refusal_score"] = j["score"]
                row["harmful_refusal_reasoning"] = j["reasoning"]
                score_lines.append((f"harmful_refusal={j['score']}/2", j["reasoning"]))

            # ── Honesty under pressure judge ──────────────────────────────
            if "honesty_under_pressure" in dims:
                j = score_honesty_under_pressure(q, answer, client)
                row["honesty_under_pressure_score"] = j["score"]
                row["honesty_under_pressure_reasoning"] = j["reasoning"]
                score_lines.append((f"honesty_pressure={j['score']}/2", j["reasoning"]))

            # ── Premise correction judge ──────────────────────────────────
            if "premise_correction" in dims:
                j = score_premise_correction(q, answer, client)
                row["premise_correction_score"] = j["score"]
                row["premise_correction_reasoning"] = j["reasoning"]
                score_lines.append((f"premise_correction={j['score']}/2", j["reasoning"]))

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize label: allow only alphanumerics, hyphens, underscores to prevent
    # path traversal (e.g. --label "../../etc/passwd") in the output filename.
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label) if label else ""
    label_part = f"_{safe_label}" if safe_label else ""
    out_path = os.path.join(RESULTS_DIR, f"run_{timestamp}{label_part}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Store run metadata alongside results so compare.py can surface it
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


def _print_summary(model: str, results: list) -> None:
    def _avg(key):
        vals = [r[key] for r in results if key in r]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    corr_avg, corr_n = _avg("correctness_score")
    nf_avg, nf_n     = _avg("not_found_score")
    th_avg, th_n     = _avg("temporal_hedging_score")
    search_rate      = sum(1 for r in results if r["searched"]) / len(results)
    # Citation rate only over cases where a citation was expected
    expected_cite    = [r for r in results if r.get("citation_expected", True)]
    citation_rate    = sum(1 for r in expected_cite if r["citation_present"]) / len(expected_cite) if expected_cite else 0
    search_ok_rate   = sum(1 for r in results if r["search_behavior_ok"]) / len(results)
    avg_searches     = sum(r["search_count"] for r in results) / len(results)

    print(f"\n  {'─'*55}")
    print(f"  Summary — {model}")
    print(f"  {'─'*55}")
    if corr_avg is not None:
        print(f"  Correctness (avg):       {corr_avg:.2f}/2.00  (n={corr_n})")
    if nf_avg is not None:
        print(f"  Not-found acknowledgment:{nf_avg:>6.2f}/2.00  (n={nf_n})")
    if th_avg is not None:
        print(f"  Temporal hedging (avg):  {th_avg:>6.2f}/2.00  (n={th_n})")
    print(f"  Search rate:             {search_rate:.0%}  ({sum(1 for r in results if r['searched'])}/{len(results)} cases)")
    print(f"  Search behavior match:   {search_ok_rate:.0%}")
    print(f"  Citation rate (expected):{citation_rate:>5.0%}")
    print(f"  Avg searches/question:   {avg_searches:.1f}")
    print(f"  Total searches:          {sum(r['search_count'] for r in results)}")

    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"correct": [], "not_found": [], "temporal": [], "searches": 0}
        if "correctness_score" in r:
            cats[c]["correct"].append(r["correctness_score"])
        if "not_found_score" in r:
            cats[c]["not_found"].append(r["not_found_score"])
        if "temporal_hedging_score" in r:
            cats[c]["temporal"].append(r["temporal_hedging_score"])
        cats[c]["searches"] += r["search_count"]

    print(f"\n  {'Category':<22} {'Correct':>8} {'NotFound':>9} {'Temporal':>9} {'Searches':>9}")
    print(f"  {'─'*62}")
    for cat, data in cats.items():
        c_str = f"{sum(data['correct'])/len(data['correct']):.2f}" if data["correct"] else "—"
        nf_str = f"{sum(data['not_found'])/len(data['not_found']):.2f}" if data["not_found"] else "—"
        th_str = f"{sum(data['temporal'])/len(data['temporal']):.2f}" if data["temporal"] else "—"
        print(f"  {cat:<22} {c_str:>8} {nf_str:>9} {th_str:>9} {data['searches']:>9}")


def _print_comparison(all_results: dict, models: list) -> None:
    col = 22
    print(f"\n{'=' * 70}")
    print("  Model comparison")
    print(f"{'=' * 70}")
    short = [m.replace("claude-", "").replace("-20251001", "") for m in models]
    print(f"  {'Metric':<30}" + "".join(f"{s:>{col}}" for s in short))
    print(f"  {'─'*60}")

    def avg(results, key):
        vals = [r[key] for r in results if key in r]
        return f"{sum(vals)/len(vals):.2f}" if vals else "—"

    for label, key in [
        ("Correctness (avg/2)", "correctness_score"),
        ("Not-found ack (avg/2)", "not_found_score"),
        ("Temporal hedging (avg/2)", "temporal_hedging_score"),
    ]:
        print(f"  {label:<30}" + "".join(f"{avg(all_results[m], key):>{col}}" for m in models))

    for label, key in [("Search rate", "searched"), ("Citation rate", "citation_present")]:
        print(f"  {label:<30}" + "".join(
            f"{sum(1 for r in all_results[m] if r[key])/len(all_results[m]):>{col}.0%}" for m in models
        ))

    total = [("Total searches", lambda m: sum(r["search_count"] for r in all_results[m]))]
    for label, fn in total:
        print(f"  {label:<30}" + "".join(f"{fn(m):>{col}}" for m in models))


def main():
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
        help="Short name for this run, e.g. 'baseline' or 'v2-temporal-hedging'. "
             "Included in the filename and stored in result metadata.",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case IDs to run, e.g. '27,28,29,30'. Runs all cases if omitted.",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    case_ids = [int(x.strip()) for x in args.cases.split(",") if x.strip()] if args.cases else None
    run_eval(models, verbose=args.verbose, label=args.label, case_ids=case_ids)


if __name__ == "__main__":
    main()
