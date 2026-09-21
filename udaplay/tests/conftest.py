"""Offline fixtures: a deterministic embedding function, a scriptable fake LLM and a fake Tavily client."""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import AIMessage  # noqa: E402
from vector_store import GAMES_COLLECTION, MEMORY_COLLECTION, VectorStoreManager  # noqa: E402

DIMS = 512


class HashEmbedding(EmbeddingFunction[Documents]):
    """Bag-of-words hashing embeddings: no network, fully deterministic."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * DIMS
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "udaplay-hash-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding()


class FakeLLM:
    """Scriptable stand-in for lib.LLM.

    ``chat_script``: AIMessages returned by successive ``invoke`` calls.
    ``parsed``: {schema class: object or callable(messages) -> object} for ``parse``.
    """

    def __init__(self, chat_script=None, parsed=None):
        self.chat_script = list(chat_script or [])
        self.parsed = parsed or {}
        self.invocations: list = []
        self.parse_calls: list = []

    def invoke(self, messages, tools=None):
        self.invocations.append(list(messages))
        return self.chat_script.pop(0) if self.chat_script else AIMessage("(no more scripted replies)")

    def parse(self, messages, schema):
        self.parse_calls.append((schema, list(messages)))
        result = self.parsed[schema]
        return result(messages) if callable(result) else result


class FakeTavily:
    def __init__(self, results=None, answer="", error: Exception | None = None):
        self.results = results or []
        self.answer = answer
        self.error = error
        self.queries: list[str] = []

    def search(self, query, **kwargs):
        self.queries.append(query)
        if self.error:
            raise self.error
        return {"answer": self.answer, "results": self.results}


@pytest.fixture
def stores(tmp_path):
    client = VectorStoreManager.persistent_client(tmp_path / "chroma")
    games = VectorStoreManager(client, GAMES_COLLECTION, HashEmbedding())
    memory = VectorStoreManager(client, MEMORY_COLLECTION, HashEmbedding())
    games.load_games(ROOT / "games")
    return games, memory
