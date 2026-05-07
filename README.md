# wiki-qa

A Wikipedia QA system powered by Claude. Claude uses a `search_wikipedia` tool to look up facts before answering, grounding responses in retrieved sources.

Built as part of an Anthropic prompt engineering take-home assignment.

## Setup

### 1. Get an Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign in or create a free account
3. Navigate to **API Keys** → **Create Key**
4. Copy the key (starts with `sk-ant-`)

### 2. Install dependencies

Requires Python 3.11+.

```bash
cd wiki-qa
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set your API key

```bash
cp .env.example .env
# Edit .env and paste your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Or export it directly:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Ask a question

```bash
python wiki_qa.py "Who directed the highest-grossing film of 1997?"
```

Output shows which Wikipedia articles were searched and the grounded answer with source citation.

### Demo mode (5 example questions)

```bash
python wiki_qa.py --demo
```

### Choose a model

```bash
python wiki_qa.py --model claude-haiku-4-5-20251001 "What is the boiling point of tungsten?"
```

Available models:
- `claude-sonnet-4-6` (default — higher quality)
- `claude-haiku-4-5-20251001` (faster, lower cost)

---

## Eval suite

Runs 20 test cases across 6 categories (simple factual, multi-hop, disambiguation, time-sensitive, not-on-Wikipedia, adversarial premise). Scores correctness and calibration using an LLM judge, plus programmatic search-behavior and citation checks.

### Run on default model (Sonnet)

```bash
python eval/run_eval.py
```

### Compare both models

```bash
python eval/run_eval.py --models claude-sonnet-4-6,claude-haiku-4-5-20251001
```

Results are saved to `results/run_<timestamp>.json`.

---

## Project structure

```
wiki-qa/
├── wiki_qa.py          # Agent: system prompt, tool definition, agentic loop, CLI
├── wikipedia.py        # MediaWiki API wrapper (search + fetch extract)
├── requirements.txt
├── .env.example
├── eval/
│   ├── cases.json      # 20 hand-curated test cases
│   ├── judge.py        # LLM-as-judge scoring (correctness + calibration)
│   └── run_eval.py     # Eval runner — runs all cases, prints summary
├── results/            # Eval run outputs (gitignored)
└── docs/
    └── rationale.md    # Design rationale (prompt choices, eval design, findings)
```

## Design rationale

See [`docs/rationale.md`](docs/rationale.md) for a full write-up of prompt engineering choices, eval design decisions, findings, and iteration history.
