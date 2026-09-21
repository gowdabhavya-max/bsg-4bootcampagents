"""The three UdaPlay tools (Part 2): retrieve_game, evaluate_retrieval, game_web_search."""
from __future__ import annotations

import json
from typing import Any

from lib import LLM, SystemMessage, UserMessage, Tool, tool
from schemas import EvaluationReport
from vector_store import VectorStoreManager

MAX_WEB_CHARS = 900  # per result, keeps the answer prompt small

JUDGE_PROMPT = """\
You are a strict evaluator for a video-game research assistant. Decide whether the retrieved \
documents contain enough information to answer the user's question.

Rules:
- Judge the specific question: a document about the right game but the wrong platform, year or \
company does NOT answer it.
- If the question asks about the current or future state of things ("right now", "latest", \
"upcoming", "working on"), static records and old memory entries are NOT enough: useful=false.
- If the documents are empty or unrelated, useful=false.
- confidence is 0.0-1.0: how sure you are about your useful/not-useful verdict.
- Explain your reasoning in detail and list any missing information.
"""


def build_tools(
    games_store: VectorStoreManager,
    memory_store: VectorStoreManager | None,
    llm: LLM,
    tavily_client: Any,
    n_results: int = 5,
) -> list[Tool]:
    """Create the tools bound to the given stores, model and Tavily client."""

    @tool
    def retrieve_game(query: str) -> list:
        """Semantic search: finds the most relevant results in the internal knowledge base
        (the game vector DB plus facts remembered from earlier web searches).

        Args:
            query: a question about the game industry.

        Returns a list of results, closest first. Each element contains:
        - source: 'internal' (game record) or 'memory' (fact learned earlier)
        - Platform, Name, YearOfRelease, Genre, Publisher, Developer, Description (internal results)
        - topic, fact, source_url, saved_at (memory results)
        - distance: semantic distance, lower means more similar
        """
        results: list[dict[str, Any]] = []
        for hit in games_store.query(query, n_results):
            results.append({"id": hit["id"], **hit["metadata"], "distance": round(hit["distance"], 4)})
        if memory_store is not None:
            for hit in memory_store.query(query, n_results):
                meta = hit["metadata"]
                results.append(
                    {
                        "id": hit["id"],
                        "source": "memory",
                        "topic": meta.get("topic"),
                        "fact": meta.get("fact"),
                        "source_url": meta.get("source_url"),
                        "saved_at": meta.get("saved_at"),
                        "distance": round(hit["distance"], 4),
                    }
                )
        results.sort(key=lambda r: r["distance"])
        return results[:n_results]

    @tool
    def evaluate_retrieval(question: str, retrieved_docs: list) -> dict:
        """Based on the user's question and the retrieved documents, analyse whether the
        documents are usable to answer that question (LLM-as-judge).

        Args:
            question: original question from the user.
            retrieved_docs: documents most similar to the question, as returned by retrieve_game.

        Returns an evaluation with:
        - useful: whether the documents are enough to answer the question
        - confidence: 0.0-1.0 confidence in that verdict
        - description: explanation of the evaluation result
        - missing_information: what is still needed, if anything
        """
        docs = retrieved_docs
        if isinstance(docs, str):  # some models pass the list as a JSON string
            try:
                docs = json.loads(docs)
            except json.JSONDecodeError:
                docs = [docs]
        if not docs:
            return EvaluationReport(
                useful=False,
                confidence=1.0,
                description="The internal knowledge base returned no documents.",
                missing_information=["Any information about the question"],
            ).model_dump()
        report = llm.parse(
            [
                SystemMessage(JUDGE_PROMPT),
                UserMessage(
                    f"Question: {question}\n\nRetrieved documents:\n{json.dumps(docs, ensure_ascii=False, indent=2)}"
                ),
            ],
            EvaluationReport,
        )
        report.confidence = report.clamped_confidence()
        return report.model_dump()

    @tool
    def game_web_search(question: str) -> dict:
        """Web search (Tavily): use when the internal knowledge base cannot answer the
        question, e.g. for recent news, unknown games or missing details.

        Args:
            question: a question about the video game industry.

        Returns a dict with:
        - summary: a short synthesized answer from the search engine (may be empty)
        - results: list of {title, url, content, score}
        """
        response = tavily_client.search(
            query=question, search_depth="basic", max_results=5, include_answer=True
        )
        return {
            "summary": response.get("answer") or "",
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": (r.get("content") or "")[:MAX_WEB_CHARS],
                    "score": r.get("score"),
                }
                for r in response.get("results", [])
            ],
        }

    return [retrieve_game, evaluate_retrieval, game_web_search]
