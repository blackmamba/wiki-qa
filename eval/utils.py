"""Shared utilities for the Wikipedia QA eval suite."""

from typing import Optional


def safe_avg(values: list[float]) -> Optional[float]:
    """Return the mean of *values*, or None if the list is empty.

    Avoids ZeroDivisionError when a metric has no applicable cases in a run.
    """
    return sum(values) / len(values) if values else None


def short_model_name(model_id: str) -> str:
    """Strip common Claude model ID prefixes/suffixes for compact display.

    Example: "claude-haiku-4-5-20251001" -> "haiku-4-5"
    """
    return model_id.replace("claude-", "").replace("-20251001", "")


def citation_rate(results: list[dict], expected_only: bool = True) -> float:
    """Return the fraction of results that include a Wikipedia citation.

    Args:
        results: List of per-case result dicts from the eval runner.
        expected_only: When True (default), only include cases where
            ``citation_expected`` is True in the denominator.

    Returns:
        A float in [0, 1], or 0.0 if the filtered list is empty.
    """
    if expected_only:
        results = [r for r in results if r.get("citation_expected", True)]
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("citation_present")) / len(results)
