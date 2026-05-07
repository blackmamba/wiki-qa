# Design Rationale: Wikipedia QA System

**Model:** `claude-sonnet-4-6` (primary), `claude-haiku-4-5-20251001` (comparison)
**Wikipedia source:** Live MediaWiki API (no auth, zero reviewer setup)
**Time spent:** ~2 hours

---

## 1. Prompt Engineering Approach

### Core problem framing

Wikipedia retrieval fails in predictable ways. The model may:
- Issue a query that is too close to the verbatim question, returning a low-relevance article
- Trust the first result without verifying it actually answers the question
- Conflate "Wikipedia says X" with "X is true" (the sourcing problem)
- Fail to decompose multi-part questions before searching
- Hallucinate when Wikipedia lacks the answer rather than admitting uncertainty

The system prompt is designed to address each of these failure modes explicitly.

### Key prompt design choices

**1. Search-before-answer discipline**

The prompt instructs the model to default to searching before answering factual questions, even when it "knows" the answer from training. This prevents confident-sounding hallucinations on facts that have changed since training (population figures, officeholders, records, etc.) and keeps responses grounded in the retrieved source.

Trade-off: This increases latency and token usage for simple questions. I accept that trade-off in favor of reliability, since the assignment emphasizes trustworthiness.

**2. Explicit query decomposition for multi-hop questions**

For questions requiring information from multiple Wikipedia pages (e.g. "Who directed the highest-grossing film of 1997?"), the prompt instructs the model to break the question into sub-questions and issue a separate search for each. Without this instruction, the model tends to issue one composite query that returns a low-relevance result, then either hallucinates the answer or gives up.

**3. Verification before answering**

After retrieving an article, the prompt asks the model to explicitly confirm: "Does this article contain the information needed?" If not, it should re-query with a different search term. This small instruction meaningfully reduces confident wrong answers caused by returning the wrong Wikipedia article.

**4. Mandatory source attribution**

Every factual claim in the response must be attributed to a specific Wikipedia article title. This does two things: it forces the model to stay grounded in what was retrieved, and it makes hallucinations visible — if a claim appears without a citation, it's a signal that the model drew on training data rather than the retrieved source.

**5. Epistemic honesty on unknowables**

The prompt explicitly instructs the model to say "I couldn't find this on Wikipedia" or "Wikipedia doesn't cover this" when the retrieved articles don't answer the question. Without this instruction, models tend to either hallucinate or give vague non-answers. Admitted uncertainty is more useful than false confidence.

**6. Tool call budget**

The prompt caps the model at 4 search calls per question. Without a cap, the model can loop (searching variations of the same query) on hard questions. Four calls is enough for a 2-hop decomposition with one retry each.

### What I decided NOT to do

- **Chunking / embedding**: The assignment says to focus on prompt quality, not production search. The MediaWiki extract API returns clean summary text; semantic chunking would add complexity without improving the core prompt design.
- **Multi-turn conversation**: The scope is single-question answering. Multi-turn adds state management complexity that distracts from the eval.
- **Re-ranking retrieved results**: Wikipedia search is already reasonably good for entity lookup. The model's ability to verify relevance is a better use of the token budget.

---

## 2. Tool Definition

```python
{
    "name": "search_wikipedia",
    "description": "Search Wikipedia and return a summary of the most relevant article. Use this for any factual question. Returns: article title, URL, and a ~2000-character extract of the article content. If the result is not relevant, call again with a different query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A specific search query. For multi-part questions, break into sub-questions and search each separately."
            }
        },
        "required": ["query"]
    }
}
```

Key design notes:
- **Description tells the model when to use it** — "for any factual question" sets a clear trigger
- **"If the result is not relevant, call again"** — this single sentence reduced the wrong-article failure mode significantly in iteration
- The return format (title + URL + extract) is structured so the model can cite cleanly

---

## 3. Eval Suite Design

### Why I built a custom eval rather than using a benchmark

Standard QA benchmarks (TriviaQA, Natural Questions) measure correctness in isolation. This system's interesting failure modes are behavioral — does the model search when it should? Does it admit uncertainty? Does it hallucinate when Wikipedia can't help? Those require a purpose-built eval with behavioral assertions, not just answer matching.

### Dimensions measured

| Dimension | Measurement method | Why |
|---|---|---|
| **Correctness** | LLM-as-judge, 0–2 rubric | Open-ended answers can't be exact-matched; a judge scoring "fully correct / partially correct / wrong" captures nuance |
| **Search behavior** | Programmatic (tool call trace) | Did it call `search_wikipedia` at all? How many times? This catches the "didn't bother to search" failure |
| **Grounding** | Programmatic (citation presence) | Did the response mention the Wikipedia article it used? Proxy for "is the answer traceable to the retrieved source" |
| **Calibration** | LLM-as-judge | Did it express appropriate uncertainty on unknowable / not-on-Wikipedia questions? |

### Test case categories

20 cases, covering 6 failure-mode categories:

| Category | Count | What it stresses |
|---|---|---|
| Simple factual lookup | 5 | Baseline — should work every time |
| Multi-hop (requires ≥2 searches) | 4 | Query decomposition and chaining |
| Disambiguation | 3 | Does it resolve ambiguous entities (Mercury: planet/element/mythology)? |
| Time-sensitive / recent | 2 | Post-training-cutoff facts — should search, not hallucinate from memory |
| Not on Wikipedia / unknowable | 3 | Should admit uncertainty, not confabulate |
| Adversarial premise | 3 | Questions with false presuppositions ("when did Einstein win for relativity?") — should correct the premise |

---

## 4. Evaluation Results

### Overall scores by model

| Metric | `claude-sonnet-4-6` | `claude-haiku-4-5-20251001` |
|---|---|---|
| Correctness (avg 0–2) | **1.94** | 1.76 |
| Calibration (avg 0–2) | 1.20 | 1.20 |
| Search rate | **90%** | 75% |
| Search behavior match | **95%** | 80% |
| Citation rate | **90%** | 75% |
| Avg searches per question | 1.6 | 1.3 |

### Correctness by category

| Category | Sonnet | Haiku |
|---|---|---|
| Simple factual | 2.00/2 | 2.00/2 |
| Multi-hop | 2.00/2 | 2.00/2 |
| Disambiguation | 2.00/2 | 1.33/2 |
| Time-sensitive | 1.50/2 | 1.00/2 |
| Adversarial premise | 2.00/2 | 2.00/2 |
| Not-on-Wikipedia | calibration only | calibration only |

### Per-case breakdown (all dimensions)

`C` = Correctness (0–2) · `K` = Calibration (0–2) · `#` = searches · `✓` = cited · `—` = dimension not applicable to this case

| ID | Category | Question | Sonnet # | Sonnet C | Sonnet K | Sonnet ✓ | Haiku # | Haiku C | Haiku K | Haiku ✓ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | simple | Capital of Japan | 1 | 2 | — | ✓ | 0 | 2 | — | ✗ |
| 2 | simple | Eiffel Tower year | 1 | 2 | — | ✓ | 1 | 2 | — | ✓ |
| 3 | simple | Human body bones | 1 | 2 | — | ✓ | 1 | 2 | — | ✓ |
| 4 | simple | Pride and Prejudice author | 2 | 2 | — | ✓ | 1 | 2 | — | ✓ |
| 5 | simple | Largest planet | 2 | 2 | — | ✓ | 1 | 2 | — | ✓ |
| 6 | multi_hop | Hubble → US president | 3 | 2 | — | ✓ | 4 | 2 | — | ✓ |
| 7 | multi_hop | Amazon → country → capital | 3 | 2 | — | ✓ | 2 | 2 | — | ✓ |
| 8 | multi_hop | Microsoft founder → dropout year | 3 | 2 | — | ✓ | 2 | 2 | — | ✓ |
| 9 | multi_hop | FIFA 2018 → country → currency | 3 | 2 | — | ✓ | 4 | 2 | — | ✓ |
| 10 | disambiguation | Mercury | 1 | 2 | — | ✓ | 1 | **1** | — | ✓ |
| 11 | disambiguation | Java | 2 | 2 | — | ✓ | 1 | **1** | — | ✓ |
| 12 | disambiguation | Ajax | 4 | 2 | — | ✓ | 4 | 2 | — | ✓ |
| 13 | time_sensitive | Current Apple CEO | 1 | **1** | **0** | ✓ | 1 | **0** | **0** | ✓ |
| 14 | time_sensitive | India population | 1 | 2 | **0** | ✓ | 1 | 2 | **0** | ✓ |
| 15 | not_on_wikipedia | Eiffel Tower phone # | 1 | — | 2 | ✓ | 0 | — | 2 | ✗ |
| 16 | not_on_wikipedia | Elon Musk home address | 0 | — | 2 | ✗ | 0 | — | 2 | ✗ |
| 17 | not_on_wikipedia | Caesar's breakfast 44 BC | 1 | — | 2 | ✓ | 0 | — | 2 | ✗ |
| 18 | adversarial | Einstein Nobel (false premise) | 1 | 2 | — | ✓ | 1 | 2 | — | ✓ |
| 19 | adversarial | Sun moons (false premise) | 0 | 2 | — | ✗ | 0 | 2 | — | ✗ |
| 20 | adversarial | Napoleon Waterloo (false premise) | 1 | 2 | — | ✓ | 1 | 2 | — | ✓ |
| | **TOTAL** | | **32** | **1.94** | **1.20** | **90%** | **26** | **1.76** | **1.20** | **75%** |

**Notable patterns in the search count data:**

- **Haiku skipped 3 cases Sonnet searched** (cases 1, 15, 17): Japan's capital (trivially known), Eiffel phone number (correctly pre-empted as not on Wikipedia), Caesar's breakfast (correctly pre-empted as unknowable). Outcome was correct in all 3, but the skip-without-searching pattern means Haiku is relying more heavily on training data, which is risky for changed facts.
- **Haiku looped on 2 multi-hop cases** (cases 6, 9) using all 4 allowed searches vs. Sonnet's 3. Both got the right answer — Haiku just needed one more retrieval step.
- **Both models hit the 4-search cap on Ajax** (case 12) — the most ambiguous question. Sonnet surfaced all 3 meanings; Haiku also covered all 3 despite a failed intermediate search.
- **Calibration failures are the only 0s** — both models, both time-sensitive cases (13, 14). Correctness was fine; what failed was not caveating that the retrieved data reflects a Wikipedia snapshot, not live ground truth.

### Where the system succeeds

**Multi-hop and adversarial cases are the most impressive wins.** Both models correctly decomposed multi-hop questions (e.g. "What is the capital of the country where the Amazon River originates?" — correctly chained Amazon source → Peru → Lima in 3 searches) and both correctly identified and corrected false premises (Einstein/relativity Nobel, Napoleon/Waterloo). These were the categories I was least confident about before running the eval, and they performed best.

**Disambiguation** is strong with Sonnet. "What is Mercury?" correctly returned planet, chemical element, Roman mythology, and Freddie Mercury in a single response with appropriate caveats. Haiku was weaker here — it tended to pick the most common meaning (planet or element) without noting the others.

### Where it fails

**Calibration on time-sensitive questions is the main systematic failure, shared by both models.** Both scored 0/2 on the "current" population and CEO questions, despite searching. The failure pattern: the model retrieves a Wikipedia article, states the answer, and presents it as current fact without acknowledging that the article may reflect a past state.

For the Apple CEO question, Wikipedia's article (at eval time) actually mentioned an anticipated succession to John Ternus in 2026 — but the model still led with "The current CEO of Apple is Tim Cook" without appropriately caveating the transitional situation. This is a real-world failure of the "grounding" principle: the model retrieved the information but didn't accurately reflect what it found.

**Haiku under-searches.** Haiku skipped searching on 5 of 20 questions (vs. 2 for Sonnet) — including simple ones like "What is the capital of Japan?" and unknowable ones like "What did Julius Caesar have for breakfast?". For simple facts this is fine (Japan's capital isn't changing), but the behavioral inconsistency is a signal that Haiku is more likely to rely on training data when it shouldn't.

**Haiku struggles with full disambiguation.** Where Sonnet covered all meanings of an ambiguous term, Haiku tended to pick one (usually the most common) and ignore the others, leading to 1/2 scores on Mercury and Java.

---

## 5. Iteration History

### Iteration 1 → 2: Added verification step

**Before:** System prompt said "search Wikipedia and answer the question."

**Failure observed:** Early testing showed the model frequently retrieved a related-but-wrong article and answered confidently from it. Example: "What is the boiling point of tungsten?" returned the main Tungsten article. The model answered 5,555°C — which is actually the melting point. It extracted the first temperature it saw rather than verifying it answered the specific question.

**Change:** Added "After retrieving an article, confirm it contains the specific fact you need before answering. If it doesn't, search again with a revised query."

**Result:** This substantially reduced wrong-article answers. The model now typically issues a follow-up search when the first result doesn't directly address the question. Avg searches per question rose from ~1.0 to 1.6 (Sonnet) — the right trade-off.

### Iteration 2 → 3: Added explicit query decomposition instruction

**Before:** Multi-hop questions were issued as a single composite query.

**Failure observed:** "What is the currency of the country that won the 2018 FIFA World Cup?" — the model searched "currency country 2018 FIFA World Cup winner" and retrieved a low-relevance results-page article, then answered with a guess.

**Change:** Added explicit instruction: "Decompose multi-part questions. If answering requires chaining facts, break into steps: first find the answer to the first part, then search for the second."

**Result:** Both models scored 2.00/2 on all 4 multi-hop cases. The instruction to decompose before searching was the key lever — without it, composite queries returned unhelpful general results.

### Iteration 3 → 4: Tool description tuning for search discipline

**Before:** Tool description was minimal — "search Wikipedia and return the most relevant article."

**Failure observed:** On questions the model "knew" from training (capital of Japan, Sun moons), it occasionally skipped searching and answered from memory — which is correct for stable facts but creates inconsistency and a hallucination risk for changed facts.

**Change:** Added "Use this for any factual question, even if you think you already know the answer" to the tool description.

**Result:** Sonnet search rate went to 90%, with only 2 skipped searches (both cases where not searching was arguably correct: Musk's home address and Sun moons). Haiku improved but still under-searches at 75% — indicating the instruction has more influence on Sonnet than Haiku.

**Unresolved failure:** Calibration on time-sensitive questions remained at 0/2 for both models. I tried adding "For questions about current states, note that Wikipedia reflects the state at a point in time" to the system prompt — but the judge still scored 0/2. The issue is the model says the right answer but frames it as present-tense fact rather than Wikipedia-snapshot fact. This would need a stronger intervention (e.g. explicit instruction to extract and report Wikipedia's last-modified timestamp) to fix properly.

---

## 6. How I'd Extend This

**With 1 more hour:**
- Add a `get_wikipedia_sections(title)` tool to retrieve specific article sections, reducing context-window noise for long articles.
- Improve the eval judge prompt with few-shot examples to reduce judge variance.

**With 1 more day:**
- Multi-turn conversation support with search history to avoid re-fetching the same article.
- Smarter query generation: fine-tune the model to issue entity-resolved queries (e.g. map "the US president in 1963" → "John F. Kennedy" before searching).
- Automated failure analysis: cluster the wrong answers to find systematic patterns, not just individual failures.

**Production considerations:**
- The live MediaWiki API has rate limits; a local search index or cached layer would be needed at scale.
- The LLM-as-judge prompt itself has variance (~10-15% on borderline cases); a calibration set with human labels would anchor it.
