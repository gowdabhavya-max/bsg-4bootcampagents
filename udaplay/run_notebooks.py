"""Execute both solution notebooks top to bottom and save them with their outputs.

    python run_notebooks.py

Needs a real ``.env`` (OPENAI_API_KEY, TAVILY_API_KEY, OPENAI_BASE_URL). It starts from a clean
``chromadb/`` folder so the saved run shows the full story: games embedded, a web search that is
saved to long-term memory, and the same question answered from memory the second time.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from config import PROJECT_DIR, Settings

NOTEBOOKS = ["Udaplay_01_solution_project.ipynb", "Udaplay_02_solution_project.ipynb"]


def main() -> int:
    try:
        settings = Settings.from_env()  # fail early, before touching anything, if a key is missing
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1

    if settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)
        print(f"Removed old vector DB at {settings.chroma_path}")

    for name in NOTEBOOKS:
        path = PROJECT_DIR / name
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_DIR)}})
        print(f"Running {name} ...")
        try:
            client.execute()
        except CellExecutionError as exc:
            nbformat.write(notebook, path)  # keep the partial output for debugging
            print(f"FAILED in {name}:\n{exc}")
            return 1
        nbformat.write(notebook, path)
        print(f"  saved {name} with outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
