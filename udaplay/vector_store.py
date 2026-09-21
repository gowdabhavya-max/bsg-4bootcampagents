"""Reusable ChromaDB vector store manager (Part 1 - RAG pipeline)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.utils import embedding_functions

from config import Settings

GAMES_COLLECTION = "udaplay"
MEMORY_COLLECTION = "udaplay_memory"


def make_embedding_function(settings: Settings):
    """OpenAI embeddings through the configured (possibly proxied) endpoint."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        api_base=settings.openai_base_url,
        model_name=settings.embedding_model,
    )


def game_to_document(game: dict[str, Any]) -> str:
    """The text that gets embedded for one game record."""
    developer = f" Developer: {game['Developer']}." if game.get("Developer") else ""
    return (
        f"{game['Name']} [{game['Platform']}] ({game['YearOfRelease']}) - {game['Genre']}."
        f"{developer} Publisher: {game['Publisher']}. {game['Description']}"
    )


class VectorStoreManager:
    """One named Chroma collection with add / semantic-search helpers.

    Cosine distance is used so ``distance`` is comparable across queries
    (0 = identical, 2 = opposite).
    """

    def __init__(self, client: chromadb.api.ClientAPI, name: str, embedding_function: Any):
        self.client = client
        self.name = name
        self.collection = client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def persistent_client(cls, path: str | Path) -> chromadb.api.ClientAPI:
        return chromadb.PersistentClient(path=str(path))

    # ---- writing -------------------------------------------------------
    def add_documents(
        self, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[dict[str, Any]]
    ) -> None:
        """Insert or overwrite documents (idempotent, safe to re-run)."""
        if ids:
            self.collection.upsert(ids=list(ids), documents=list(documents), metadatas=list(metadatas))

    def load_games(self, games_dir: str | Path) -> int:
        """Embed every ``*.json`` game record in ``games_dir``. Returns the number loaded."""
        ids, docs, metas = [], [], []
        for path in sorted(Path(games_dir).glob("*.json")):
            game = json.loads(path.read_text(encoding="utf-8"))
            ids.append(path.stem)
            docs.append(game_to_document(game))
            metas.append({**game, "source": "internal"})
        self.add_documents(ids, docs, metas)
        return len(ids)

    # ---- reading -------------------------------------------------------
    def query(self, text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Semantic search. Returns dicts with ``id``, ``document``, ``metadata`` and ``distance``."""
        total = self.count()
        if total == 0:
            return []
        result = self.collection.query(query_texts=[text], n_results=min(n_results, total))
        return [
            {"id": id_, "document": doc, "metadata": meta, "distance": dist}
            for id_, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the (empty) collection."""
        embedding_function = self.collection._embedding_function  # noqa: SLF001 - reuse the same function
        self.client.delete_collection(self.name)
        self.collection = self.client.create_collection(
            name=self.name, embedding_function=embedding_function, metadata={"hnsw:space": "cosine"}
        )
