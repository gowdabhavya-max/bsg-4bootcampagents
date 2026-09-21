# UdaPlay - video game research agent

UdaPlay answers natural-language questions about video games ("Who developed FIFA 21?", "When was God of War Ragnarök released?", "What is Rockstar Games working on right now?").

It uses a **two-tier retrieval system**:

1. **RAG** over a local dataset of games (ChromaDB + OpenAI embeddings)
2. **Web search** (Tavily) when the internal knowledge is missing or the retrieval is judged low quality

Facts learned from the web are **parsed and persisted in long-term memory**, so the same question is answered locally next time. Answers are structured reports that **cite their sources** and state a **confidence level**.

## How it works

```
                         ┌──────────────┐    useful & confident    ┌────────┐
question ─► retrieve_game ─► evaluate_retrieval ───────────────────► answer ─► report
              (Chroma:                │ otherwise                      ▲
           games + memory)            ▼                                │
                              game_web_search ─► remember (parse & persist)
```

| Piece | File | What it does |
|---|---|---|
| Vector store manager | [`vector_store.py`](vector_store.py) | Reusable Chroma wrapper: embed & upsert JSON games, semantic search |
| Tools | [`tools.py`](tools.py) | `retrieve_game`, `evaluate_retrieval` (LLM-as-judge), `game_web_search` |
| Agent | [`lib/agents.py`](lib/agents.py) | Tool-calling agent built on a state machine, with per-session short-term memory |
| Workflow | [`workflow.py`](workflow.py) | Explicit state machine: tools are pre-defined nodes, deterministic fallback routing |
| Long-term memory | [`long_term_memory.py`](long_term_memory.py) | Persists parsed web facts in a second Chroma collection |
| Report | [`report.py`](report.py), [`schemas.py`](schemas.py) | Structured `UdaPlayReport` rendered as Markdown |
| Toolkit | [`lib/`](lib) | Messages, `@tool`, LLM wrapper, `StateMachine`, memory |

### Design notes

- **Confidence-based fallback.** The judge returns `useful` and a `confidence` (0-1). The workflow searches the web unless the docs are useful **and** confidence ≥ 0.6 (`confidence_threshold`). Questions about the current state of things ("right now", "latest") are always sent to the web.
- **Citations can't be invented.** Evidence is numbered (`D1` internal, `M1` memory, `W1` web); the model reports which ids it used and the report's sources are built from those, dropping unknown ids.
- **Graceful degradation.** A failed web search or memory write never crashes an answer; the confidence is capped when a needed web check could not be made, and with no evidence at all the model isn't asked to guess.
- **Cosine distance** in Chroma, so `distance` is comparable across queries. Loading games is an upsert, so it is idempotent.

## Setup

```bash
cd udaplay
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env with your keys
```

You need an OpenAI key (or the Vocareum key/base URL from the course) and a free [Tavily](https://app.tavily.com/home) key (1000 free requests).

## Run it

**Notebooks** (the project deliverables, mirroring the starter structure):

1. [`Udaplay_01_solution_project.ipynb`](Udaplay_01_solution_project.ipynb) - build the Chroma VectorDB from `games/`
2. [`Udaplay_02_solution_project.ipynb`](Udaplay_02_solution_project.ipynb) - tools, agent, structured workflow, long-term memory

**Command line:**

```bash
python main.py "Who developed FIFA 21?"                # structured workflow (default)
python main.py --mode agent "Was Mortal Kombat X released for PlayStation 5?"
python main.py --trace "What is Rockstar Games working on right now?"
python main.py                                         # interactive chat
python main.py --reload                                # re-embed the games dataset
```

Example output:

```
### Who developed FIFA 21?

FIFA 21 was developed by EA Vancouver and EA Romania and published by Electronic Arts [D1][D2].

**Confidence:** High
**Internal knowledge:** sufficient (judge confidence 95%) — ...
**Web search used:** no

**Sources**
- 📚 FIFA 21 (PlayStation 4) — game record 010
- 📚 FIFA 21 (PlayStation 5) — game record 011
```

(Actual wording varies with the model.)

## Data

`games/*.json` - 34 game records, one per game **per platform**:

```json
{
  "Name": "Gran Turismo",
  "Platform": "PlayStation 1",
  "Genre": "Racing",
  "Publisher": "Sony Computer Entertainment",
  "Developer": "Polyphony Digital",
  "Description": "A realistic racing simulator ...",
  "YearOfRelease": 1997
}
```

Add more files (any `*.json` with these fields) and run `python main.py --reload`. The `Developer` field is an addition to the starter schema so that "who developed …?" questions are answerable.

## Submitting

See [`SUBMISSION.md`](SUBMISSION.md) for producing the notebook outputs, pushing to GitHub and handing in.

## Tests

The test suite runs **offline** (no API keys): a scriptable fake LLM, fake Tavily client and deterministic hash embeddings stand in for the services; `lib.LLM` itself is tested against the real OpenAI client with a mocked HTTP transport.

```bash
python -m pytest
```

Covered: dataset validity, idempotent loading, semantic search, the three tools, fallback routing (internal vs web), memory reuse without a second web search, failure handling, citation mapping, the state machine and the agent loop.

## Limits

- Retrieval quality and the LLM judge are only exercised with fakes in the test suite; run the notebooks with real keys to see end-to-end behavior.
- The dataset is small and hand-written; web-learned facts are only as good as the pages Tavily returns (each memory keeps its source URL and save date).
