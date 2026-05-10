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

Never commit `.env` — it is gitignored. The `.env.example` file contains only a placeholder value and is safe to track.

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

30 test cases across 8 behavioral categories: simple factual, multi-hop, disambiguation, time-sensitive, not-on-Wikipedia, adversarial premise, sycophancy, honesty under pressure, and harmful refusal.

Scores 7 dimensions using an LLM-as-judge (correctness, not-found acknowledgment, temporal hedging, sycophancy, harmful refusal, honesty under pressure, premise correction), plus programmatic search-behavior and citation checks.

### Run the full suite

```bash
python eval/run_eval.py
```

### Compare both models

```bash
python eval/run_eval.py --models claude-sonnet-4-6,claude-haiku-4-5-20251001
```

### Useful flags

```bash
# Verbose: show full answers and judge reasoning per case
python eval/run_eval.py -v

# Run only specific cases (e.g. honesty/safety cases)
python eval/run_eval.py --cases 27,28,29,30

# Label the run for later comparison
python eval/run_eval.py --label v5-temporal-hedging
```

Results are saved to `results/run_<timestamp>[_label].json` (gitignored).

### Compare two runs

```bash
python eval/compare.py results/run_A.json results/run_B.json
```

Shows aggregate score deltas, per-category breakdown, and per-case regressions (▼) and improvements (▲).

---

## Project structure

```
wiki-qa/
├── wiki_qa.py          # Agent: system prompt, tool definition, agentic loop, CLI
├── wikipedia.py        # MediaWiki API wrapper (search + fetch extract)
├── requirements.txt
├── .env.example        # Placeholder only — never contains real credentials
├── eval/
│   ├── cases.json      # 30 hand-curated test cases across 8 categories
│   ├── judge.py        # LLM-as-judge scoring (7 behavioral dimensions)
│   ├── run_eval.py     # Eval runner: --models, --cases, --label, -v
│   └── compare.py      # Side-by-side diff of two result files
├── results/            # Eval run outputs (gitignored)
└── docs/
    └── rationale.md    # Design rationale: prompt choices, eval design, findings, iteration history
```

## Security

- `ANTHROPIC_API_KEY` is read from `.env` at runtime and never hardcoded
- `.env` is gitignored; `.env.example` contains only a placeholder (`sk-ant-api03-...`)
- Eval result files (`results/*.json`) are gitignored — they may contain question answers but no credentials
- Wikipedia API requests go only to `en.wikipedia.org/w/api.php` over HTTPS
- The `--label` flag is sanitized to alphanumerics, hyphens, and underscores before use in filenames

## Design rationale

See [`docs/rationale.md`](docs/rationale.md) for a full write-up of prompt engineering choices, eval design decisions, findings, and iteration history.
