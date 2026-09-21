# Submitting UdaPlay

## What gets submitted

| Rubric item | File |
|---|---|
| RAG notebook (loads, processes, embeds the game JSON into a persistent ChromaDB, semantic search demo) | [`Udaplay_01_solution_project.ipynb`](Udaplay_01_solution_project.ipynb) |
| Agent notebook (3 tools, stateful state-machine agent, ≥3 example queries with reasoning, tool usage, final answer and citations) | [`Udaplay_02_solution_project.ipynb`](Udaplay_02_solution_project.ipynb) |
| Supporting code | `vector_store.py`, `tools.py`, `workflow.py`, `app.py`, `long_term_memory.py`, `report.py`, `schemas.py`, `config.py`, `lib/` |
| Data | `games/*.json` |
| Docs / tests | `README.md`, `tests/`, `requirements.txt`, `.env.example` |

**Never submit or commit `.env`** (it holds your API keys). It is already in `.gitignore`.

## 1. Produce the notebook outputs (needs your keys)

The saved output in the notebooks is what a reviewer looks at, so run them once with real keys.

```bash
cd "C:\Users\bhavya.s.gowda\Training\Building agents work\bsg-4bootcampagents\udaplay"
..\..\.udaplay-venv\Scripts\activate           # or your own venv: pip install -r requirements.txt
copy .env.example .env                         # then edit .env: OPENAI_API_KEY, TAVILY_API_KEY, OPENAI_BASE_URL
python run_notebooks.py                        # runs both notebooks top to bottom and saves their outputs
python -m pytest                               # optional: 31 offline tests
```

Or run them by hand in Jupyter / VS Code: **Run All** on notebook 01 first, then notebook 02, then save.

Check before submitting:
- notebook 01 prints `Loaded 34 game records` and the semantic-search results;
- notebook 02 shows, for each query, the reasoning, the tool calls (`retrieve_game` → `evaluate_retrieval` → `game_web_search` when needed) and a final answer with sources;
- no cell shows a traceback, and no key appears in any output.

## 2. Push to GitHub

The repository already exists and is connected: `https://github.com/gowdabhavya-max/bsg-4bootcampagents` (branch `main`).

```bash
cd "C:\Users\bhavya.s.gowda\Training\Building agents work\bsg-4bootcampagents"
git status                                  # expect: README.md modified; udaplay/ untracked; NO .env listed
git add README.md udaplay
git status                                  # double-check that .env and chromadb/ are not staged
git commit -m "Add UdaPlay: RAG pipeline and research agent"
git pull --rebase origin main               # GitHub already has a newer commit (the Beaver's Choice project)
git push origin main
```

The `pull --rebase` step matters: the remote `main` already contains another project, so a plain `git push` is rejected with "fetch first". The two projects live in separate folders and the rebase applies cleanly.

If `git push` asks you to sign in, use your GitHub account (a browser window or a personal access token as the password).

Verify on GitHub: open the repo, open `udaplay/`, and confirm both notebooks render **with their outputs** and that `.env` is not there.

## 3. Hand it in

In the Udacity **Submit Project** step, provide the GitHub repository URL (or upload a zip, see below).

- Repository URL: `https://github.com/gowdabhavya-max/bsg-4bootcampagents`
- If the reviewer needs a single project folder, point to the `udaplay/` directory: `https://github.com/gowdabhavya-max/bsg-4bootcampagents/tree/main/udaplay`

To submit a zip instead (excludes secrets, caches and the vector DB):

```bash
cd "C:\Users\bhavya.s.gowda\Training\Building agents work\bsg-4bootcampagents"
git archive --format=zip --prefix=udaplay/ -o ../udaplay-submission.zip HEAD:udaplay
```

`git archive` only includes committed files, so commit first. Because it starts from Git, `.env`, `chromadb/` and `__pycache__/` can never end up in the zip.
