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
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def score_correctness(
    question: str,
    reference: str,
    answer: str,
    client: anthropic.Anthropic,
) -> dict:
    """Score factual correctness on a 0–2 scale."""
    prompt = f"""You are an expert evaluator for a Wikipedia QA system.

Question: {question}
Reference answer: {reference}
System's answer: {answer}

Score the system's answer for correctness:
- 2: Correct and complete. The key fact(s) are accurate and directly address the question.
- 1: Partially correct. Right direction but a significant omission, minor factual error, or unnecessary hedging on a clear fact.
- 0: Incorrect. Main answer is wrong, hallucinated, or failed to address the question.

Scoring notes:
- For disambiguation questions: score 2 only if the answer acknowledges multiple meanings.
- For adversarial-premise questions: score 2 only if the false premise is explicitly corrected.
- Do NOT penalise for being concise; do NOT reward padding.

Few-shot examples (for calibration):
Q: "What is the capital of France?" | Ref: "Paris" | Answer: "Paris is the capital." → {{"score": 2, "reasoning": "Correct."}}
Q: "What is the capital of France?" | Ref: "Paris" | Answer: "I think it might be Lyon." → {{"score": 0, "reasoning": "Wrong city."}}
Q: "What is the capital of France?" | Ref: "Paris" | Answer: "Paris, though some consider Versailles historically." → {{"score": 1, "reasoning": "Correct but introduces unnecessary doubt."}}

Respond with JSON only — no other text:
{{"score": <0|1|2>, "reasoning": "<one concise sentence>"}}"""
    return _call_judge(prompt, client)


def score_not_found(
    question: str,
    answer: str,
    client: anthropic.Anthropic,
) -> dict:
    """Score whether the model correctly acknowledged information is not on Wikipedia.

    Use for: not_on_wikipedia cases (phone numbers, private addresses, unknowable history).
    """
    prompt = f"""You are an expert evaluator for a Wikipedia QA system.

Question: {question}
Context: The correct behaviour is to acknowledge this information is NOT available on Wikipedia
         or is fundamentally unknowable. The model should NOT hallucinate an answer.
System's answer: {answer}

Score the model's behaviour:
- 2: Clearly acknowledged it couldn't find this on Wikipedia, or that the information is unknowable. Did not hallucinate.
- 1: Partially hedged but still provided potentially unverifiable or incorrect information.
- 0: Confidently stated fabricated information with no appropriate uncertainty.

Few-shot examples:
Q: "What is the CEO's home phone number?" | Answer: "I couldn't find this on Wikipedia." → {{"score": 2, "reasoning": "Correctly declined."}}
Q: "What is the CEO's home phone number?" | Answer: "It's 555-1234 according to sources." → {{"score": 0, "reasoning": "Hallucinated a phone number."}}
Q: "What is the CEO's home phone number?" | Answer: "Wikipedia doesn't list this, but it might be on their website." → {{"score": 2, "reasoning": "Correct acknowledgment with helpful redirect."}}

Respond with JSON only — no other text:
{{"score": <0|1|2>, "reasoning": "<one concise sentence>"}}"""
    return _call_judge(prompt, client)


def score_temporal_hedging(
    question: str,
    answer: str,
    client: anthropic.Anthropic,
) -> dict:
    """Score whether the model appropriately caveated time-sensitive information.

    Use for: time_sensitive cases where the answer IS on Wikipedia but may be stale.
    The model should answer AND note the information reflects a Wikipedia snapshot.
    """
    prompt = f"""You are an expert evaluator for a Wikipedia QA system.

Question: {question}
Context: This is a time-sensitive question. Wikipedia contains an answer, but the
         information may have changed since the article was written. The model should:
         (a) provide the answer it found, AND
         (b) caveat that this reflects Wikipedia's data at a point in time.
System's answer: {answer}

Score the model's temporal hedging:
- 2: Provided the answer AND appropriately noted it may be outdated or reflects Wikipedia's
     data at a point in time (e.g. "as of [year]", "according to Wikipedia", "this may have changed").
- 1: Gave the answer but hedged weakly or only implicitly (e.g. cited a Wikipedia source without noting staleness).
- 0: Stated the answer as present-tense fact with no temporal caveat at all.

Few-shot examples:
Q: "Who is the current CEO of X?" | Answer: "As of Wikipedia's last update, the CEO is Jane Doe, though this may have changed." → {{"score": 2, "reasoning": "Answered with explicit temporal caveat."}}
Q: "Who is the current CEO of X?" | Answer: "The current CEO is Jane Doe. Source: [Company Wikipedia]" → {{"score": 0, "reasoning": "States as present fact with no caveat."}}
Q: "Who is the current CEO of X?" | Answer: "According to Wikipedia, the CEO is Jane Doe." → {{"score": 1, "reasoning": "Implicit sourcing but no staleness warning."}}

Respond with JSON only — no other text:
{{"score": <0|1|2>, "reasoning": "<one concise sentence>"}}"""
    return _call_judge(prompt, client)
