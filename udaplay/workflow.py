"""UdaPlay research workflow: an explicit state machine whose nodes are the three tools.

    retrieve -> evaluate --(useful & confident)--------------------> answer -> END
                         \\--(otherwise)-> web_search -> remember -> answer -> END
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from lib import END, LLM, StateMachine, SystemMessage, Tool, UserMessage
from long_term_memory import LongTermMemory
from schemas import DraftAnswer, EvaluationReport, ExtractedMemories, Source, UdaPlayReport
from vector_store import game_to_document

DEFAULT_CONFIDENCE_THRESHOLD = 0.6

ANSWER_PROMPT = """\
You are UdaPlay, a research assistant for the video game industry. Answer the user's question \
using ONLY the numbered evidence provided.

Rules:
- Cite evidence inline with its id, e.g. "... was released in 1996 [D1]".
- Combine evidence from several sources when needed, and say so if sources disagree.
- Read the question carefully (platform, year, company). If the evidence does not actually \
answer it, say what is missing instead of guessing.
- If the evidence shows the premise is false (e.g. a game was not released on that platform), say so.
- Write a clear, natural answer of a few sentences. No preamble.
- confidence: "high" if the evidence answers the question directly, "medium" if partially or \
from a single web snippet, "low" if it is thin or conflicting.
- evidence_ids: list every id you relied on.
"""

MEMORY_PROMPT = """\
Extract durable facts from these web search results that would help answer future video-game \
questions (release dates, platforms, developers, publishers, announcements, ...).

Rules:
- Only include facts explicitly stated in the results; never add outside knowledge.
- Each fact must be self-contained (name the game/company). For time-sensitive facts, begin with \
"As of {today}".
- Set source_url to the page the fact came from. Skip opinions and unrelated content.
- Return an empty list if nothing is worth remembering.
"""


class UdaPlay:
    """Answers a question by RAG first, web search second, and remembers what it learns."""

    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        memory: LongTermMemory | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.memory = memory
        self.confidence_threshold = confidence_threshold
        self.machine = self._build_machine()

    # ---- state machine -------------------------------------------------
    def _build_machine(self) -> StateMachine:
        machine = StateMachine("udaplay", max_transitions=10)
        machine.add_step("retrieve", self._retrieve, next="evaluate")
        machine.add_step("evaluate", self._evaluate, next=self._route_after_evaluation)
        machine.add_step("web_search", self._web_search, next="remember")
        machine.add_step("remember", self._remember, next="answer")
        machine.add_step("answer", self._answer, next=END)
        return machine

    def _retrieve(self, state: dict) -> dict:
        return {"retrieved": self.tools["retrieve_game"](query=state["question"])}

    def _evaluate(self, state: dict) -> dict:
        raw = self.tools["evaluate_retrieval"](question=state["question"], retrieved_docs=state["retrieved"])
        return {"evaluation": EvaluationReport(**raw)}

    def _route_after_evaluation(self, state: dict) -> str:
        evaluation: EvaluationReport = state["evaluation"]
        confident = evaluation.clamped_confidence() >= self.confidence_threshold
        return "answer" if evaluation.useful and confident else "web_search"

    def _web_search(self, state: dict) -> dict:
        try:
            response = self.tools["game_web_search"](question=state["question"])
        except Exception as exc:  # noqa: BLE001 - degrade to internal-only answer
            return {"web": {"summary": "", "results": []}, "web_error": f"{type(exc).__name__}: {exc}"}
        return {"web": response, "used_web": True}

    def _remember(self, state: dict) -> dict:
        results = state["web"]["results"]
        if self.memory is None or not results:
            return {"saved": 0}
        try:
            extracted = self.llm.parse(
                [
                    SystemMessage(MEMORY_PROMPT.format(today=date.today().isoformat())),
                    UserMessage(
                        f"Question that triggered the search: {state['question']}\n\n"
                        f"Search results:\n{json.dumps(results, ensure_ascii=False, indent=2)}"
                    ),
                ],
                ExtractedMemories,
            )
            return {"saved": self.memory.remember(extracted.entries, state["question"])}
        except Exception as exc:  # noqa: BLE001 - failing to save must not fail the answer
            return {"saved": 0, "memory_error": f"{type(exc).__name__}: {exc}"}

    def _answer(self, state: dict) -> dict:
        evidence = self._collect_evidence(state)
        if not evidence:
            report = UdaPlayReport(
                question=state["question"],
                answer="I couldn't find reliable information to answer this question.",
                confidence="low",
                used_web_search=state.get("used_web", False),
                evaluation=state.get("evaluation"),
            )
            return {"report": report}

        listing = "\n\n".join(f"[{eid}] ({item['kind']}) {item['text']}" for eid, item in evidence.items())
        draft = self.llm.parse(
            [SystemMessage(ANSWER_PROMPT), UserMessage(f"Question: {state['question']}\n\nEvidence:\n{listing}")],
            DraftAnswer,
        )
        sources = [
            Source(kind=evidence[eid]["kind"], title=evidence[eid]["title"], reference=evidence[eid]["reference"])
            for eid in dict.fromkeys(draft.evidence_ids)  # de-duplicate, keep order
            if eid in evidence  # drop ids the model invented
        ]
        confidence = draft.confidence
        if state.get("web_error") and confidence == "high":
            confidence = "medium"  # we wanted a web check and could not get one
        report = UdaPlayReport(
            question=state["question"],
            answer=draft.answer,
            confidence=confidence,
            sources=sources,
            used_web_search=state.get("used_web", False),
            saved_to_memory=state.get("saved", 0),
            evaluation=state.get("evaluation"),
        )
        return {"report": report}

    @staticmethod
    def _collect_evidence(state: dict) -> dict[str, dict[str, Any]]:
        """Number every piece of evidence so the answer can cite it and we can map ids to sources."""
        evidence: dict[str, dict[str, Any]] = {}
        counters = {"D": 0, "M": 0, "W": 0}

        def add(prefix: str, kind: str, title: str, reference: str, text: str) -> None:
            counters[prefix] += 1
            evidence[f"{prefix}{counters[prefix]}"] = {
                "kind": kind, "title": title, "reference": reference, "text": text,
            }

        for doc in state.get("retrieved", []):
            if doc.get("source") == "memory":
                add("M", "memory", doc.get("topic") or "Remembered fact", doc.get("source_url") or doc["id"],
                    f"{doc.get('fact')} (saved {doc.get('saved_at')}, from {doc.get('source_url')})")
            else:
                add("D", "internal", f"{doc.get('Name')} ({doc.get('Platform')})", f"game record {doc['id']}",
                    game_to_document(doc))

        web = state.get("web") or {}
        if web.get("summary"):
            add("W", "web", "Search summary", "Tavily search summary", web["summary"])
        for result in web.get("results", []):
            add("W", "web", result.get("title") or result["url"], result["url"], result.get("content", ""))
        return evidence

    # ---- public API ----------------------------------------------------
    def ask(self, question: str) -> UdaPlayReport:
        state = self.machine.run({"question": question})
        report: UdaPlayReport = state["report"]
        report.trace = state["trace"]
        return report
