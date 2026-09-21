"""Long-term memory: facts learned from the web, persisted in their own Chroma collection
so later questions can be answered without searching again."""
from __future__ import annotations

import hashlib
from datetime import date

from schemas import MemoryEntry
from vector_store import VectorStoreManager


class LongTermMemory:
    def __init__(self, store: VectorStoreManager):
        self.store = store

    def remember(self, entries: list[MemoryEntry], question: str = "") -> int:
        """Persist entries (deduplicated by content). Returns how many were saved."""
        ids, docs, metas = [], [], []
        for entry in entries:
            if not entry.fact.strip():
                continue
            doc = f"{entry.topic}: {entry.fact}".strip()
            ids.append("mem-" + hashlib.sha1(doc.lower().encode("utf-8")).hexdigest()[:16])
            docs.append(doc)
            metas.append(
                {
                    "source": "memory",
                    "topic": entry.topic,
                    "fact": entry.fact,
                    "entities": ", ".join(entry.entities),
                    "source_url": entry.source_url,
                    "question": question,
                    "saved_at": date.today().isoformat(),
                }
            )
        self.store.add_documents(ids, docs, metas)
        return len(ids)

    def count(self) -> int:
        return self.store.count()
