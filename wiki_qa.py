#!/usr/bin/env python3
"""Wikipedia QA agent powered by Claude."""

import argparse
import json
import sys

import anthropic
import os
from dotenv import dotenv_values
from pathlib import Path

from wikipedia import search_wikipedia

os.environ.update(dotenv_values(Path(__file__).parent / ".env"))

SYSTEM_PROMPT = """\
You are a research assistant that answers factual questions by searching Wikipedia.

## When to search

Search Wikipedia for every factual question—even if you believe you know the answer \
from training data. Your training may be outdated or contain errors. Wikipedia provides \
a citable, current source.

You do not need to search for pure logic, math, or clarifying questions.

## How to search effectively

1. Issue specific queries targeting the key entity or concept, not the full question verbatim.
   - Less effective: "who was president when berlin wall fell"
   - More effective: "Berlin Wall fall year" then "US president 1989"

2. Decompose multi-part questions. If answering requires chaining facts \
(e.g. "What is the currency of the country that won X?"), break into steps: \
first find the answer to the first part, then search for the second.

3. Verify relevance. After retrieving an article, confirm it contains the specific fact \
you need. If it doesn't, search again with a revised query. You may search up to 4 times \
per question.

4. Acknowledge gaps. If Wikipedia does not contain the answer, say so clearly: \
"I couldn't find this on Wikipedia." Do not fill in from memory or guess.

## How to answer

- Lead with the direct answer.
- Correct false premises when the question assumes something that isn't true.
- For questions about current states (current leaders, populations, records, prices): \
note that your answer reflects the Wikipedia article and may not be up to date. \
Use phrases like "as of [year]", "according to Wikipedia", or "this may have changed."
- Cite every Wikipedia article you used:
  **Source:** [Article Title](URL)
  If multiple: **Sources:** [Title 1](URL1), [Title 2](URL2)
- Keep responses concise and factual.
"""

TOOL = {
    "name": "search_wikipedia",
    "description": (
        "Search Wikipedia and return the most relevant article's title, URL, and a "
        "~2500-character extract. Use this for any factual question, even if you think "
        "you already know the answer. If the returned article is off-topic or doesn't "
        "contain the fact you need, call again with a different query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Specific search query targeting the key entity or fact. "
                    "For multi-part questions, issue separate searches for each sub-question."
                ),
            }
        },
        "required": ["query"],
    },
}

DEFAULT_MODEL = "claude-sonnet-4-6"


def run_query(question: str, model: str = DEFAULT_MODEL) -> dict:
    """Run one question through the Wikipedia QA agent. Returns a result dict."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    tool_calls = []

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[TOOL],
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    query = block.input["query"]
                    wiki_result = search_wikipedia(query)
                    tool_calls.append(
                        {"query": query, "result_title": wiki_result.get("title")}
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(wiki_result),
                        }
                    )
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            answer = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return {
                "question": question,
                "answer": answer,
                "tool_calls": tool_calls,
                "search_count": len(tool_calls),
                "searched": len(tool_calls) > 0,
                "model": model,
            }

        else:
            return {
                "question": question,
                "answer": f"Unexpected stop reason: {response.stop_reason}",
                "tool_calls": tool_calls,
                "search_count": len(tool_calls),
                "searched": len(tool_calls) > 0,
                "model": model,
            }


DEMO_QUESTIONS = [
    "Who was the US president when the Hubble Space Telescope was launched?",
    "What is the capital of the country where the Amazon River originates?",
    "When did Einstein win the Nobel Prize for the theory of relativity?",
    "What is Mercury?",
    "What is the phone number of the Eiffel Tower visitor center?",
]


def main():
    parser = argparse.ArgumentParser(description="Wikipedia QA agent powered by Claude")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        help="Claude model (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run built-in demo questions"
    )
    args = parser.parse_args()

    if args.demo:
        questions = DEMO_QUESTIONS
    elif args.question:
        questions = [args.question]
    else:
        parser.print_help()
        sys.exit(1)

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        result = run_query(q, model=args.model)
        if result["searched"]:
            searches = " → ".join(f'"{tc["query"]}"' for tc in result["tool_calls"])
            print(f"[searched {result['search_count']}x: {searches}]")
        else:
            print("[no search]")
        print(f"\nA: {result['answer']}")


if __name__ == "__main__":
    main()
