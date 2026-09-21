"""Structured data types shared by the tools, the workflow and the report."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


class EvaluationReport(BaseModel):
    """LLM-as-judge verdict on whether retrieved documents can answer a question."""

    useful: bool = Field(description="True if the documents contain enough information to answer the question.")
    confidence: float = Field(description="How confident you are in that verdict, from 0.0 to 1.0.")
    description: str = Field(description="Detailed explanation of the verdict, so an action can be chosen.")
    missing_information: list[str] = Field(
        default_factory=list, description="Facts still needed to answer fully (empty if none)."
    )

    def clamped_confidence(self) -> float:
        return max(0.0, min(1.0, self.confidence))


class MemoryEntry(BaseModel):
    """One durable fact learned from the web, worth remembering for later questions."""

    topic: str = Field(description="Short subject, e.g. a game or company name.")
    fact: str = Field(description="A self-contained, factual statement (include names, dates and platforms).")
    entities: list[str] = Field(default_factory=list, description="Games, companies or platforms mentioned.")
    source_url: str = Field(description="URL of the page the fact came from.")


class ExtractedMemories(BaseModel):
    entries: list[MemoryEntry] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    """What the model returns when writing the final answer."""

    answer: str = Field(description="Natural-language answer to the question, citing evidence ids like [D1] or [W2].")
    confidence: Confidence = Field(description="Overall confidence in the answer.")
    evidence_ids: list[str] = Field(description="Ids of the evidence items the answer relies on, e.g. ['D1', 'W2'].")


class Source(BaseModel):
    kind: Literal["internal", "memory", "web"]
    title: str
    reference: str  # game record id, memory id, or URL


class UdaPlayReport(BaseModel):
    """The final, structured result returned to the user."""

    question: str
    answer: str
    confidence: Confidence
    sources: list[Source] = Field(default_factory=list)
    used_web_search: bool = False
    saved_to_memory: int = 0
    evaluation: EvaluationReport | None = None
    trace: list[str] = Field(default_factory=list)
