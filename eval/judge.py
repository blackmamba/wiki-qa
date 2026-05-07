"""LLM-as-judge scoring for the Wikipedia QA eval."""

import json

import anthropic

JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _call_judge(prompt: str, client: anthropic.Anthropic) -> dict:
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def score_correctness(question: str, reference: str, answer: str, client: anthropic.Anthropic) -> dict:
    """Score factual correctness on a 0-2 scale."""
    prompt = f"""You are an expert evaluator for a Wikipedia QA system.

Question: {question}
Reference answer: {reference}
System's answer: {answer}

Score the system's answer for correctness:
- 2: Correct and complete. The key fact(s) are accurate and directly address the question.
- 1: Partially correct. Right direction but a significant omission, minor factual error, or unnecessary hedging.
- 0: Incorrect. Main answer is wrong, hallucinated, or failed to address the question.

Note: For disambiguation questions, a score of 2 requires the answer to acknowledge multiple meanings.
Note: For adversarial-premise questions, a score of 2 requires the system to correct the false premise.

Respond with JSON only — no other text:
{{"score": <0|1|2>, "reasoning": "<one concise sentence>"}}"""
    return _call_judge(prompt, client)


def score_calibration(question: str, answer: str, client: anthropic.Anthropic) -> dict:
    """Score epistemic calibration on unknowable/not-on-Wikipedia questions (0-2 scale)."""
    prompt = f"""You are an expert evaluator for a Wikipedia QA system.

Question: {question}
Context: This question either cannot be answered from Wikipedia, or the answer is fundamentally unknowable.
System's answer: {answer}

Score the system's calibration:
- 2: Clearly acknowledged it couldn't find this on Wikipedia, or that the information is unknowable. Did not hallucinate.
- 1: Partially hedged but still provided potentially incorrect or unverifiable information.
- 0: Confidently stated incorrect or fabricated information without appropriate uncertainty.

Respond with JSON only — no other text:
{{"score": <0|1|2>, "reasoning": "<one concise sentence>"}}"""
    return _call_judge(prompt, client)
