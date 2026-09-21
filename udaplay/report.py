"""Render a UdaPlayReport as readable Markdown."""
from __future__ import annotations

import json
from typing import Any

from schemas import UdaPlayReport

_ICON = {"internal": "📚", "memory": "🧠", "web": "🌐"}


def render_markdown(report: UdaPlayReport, show_trace: bool = False) -> str:
    lines = [
        f"### {report.question}",
        "",
        report.answer,
        "",
        f"**Confidence:** {report.confidence.capitalize()}",
    ]
    if report.evaluation is not None:
        verdict = "sufficient" if report.evaluation.useful else "insufficient"
        lines.append(
            f"**Internal knowledge:** {verdict} "
            f"(judge confidence {report.evaluation.clamped_confidence():.0%}) — {report.evaluation.description}"
        )
    lines.append(f"**Web search used:** {'yes' if report.used_web_search else 'no'}")
    if report.saved_to_memory:
        lines.append(f"**Saved to long-term memory:** {report.saved_to_memory} fact(s)")
    if report.sources:
        lines += ["", "**Sources**"]
        lines += [f"- {_ICON[s.kind]} {s.title} — {s.reference}" for s in report.sources]
    if show_trace:
        lines += ["", "_Workflow: " + " → ".join(report.trace) + "_"]
    return "\n".join(lines)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _judge_verdict(step: dict[str, Any]) -> str | None:
    """One-line summary of an ``evaluate_retrieval`` result, or None if it is not a valid verdict."""
    try:
        result = json.loads(step["output"])
        return (
            f"{'useful' if result['useful'] else 'NOT useful'} "
            f"(confidence {float(result['confidence']):.0%}) - {_shorten(result['description'], 320)}"
        )
    except (TypeError, ValueError, KeyError):
        return None


def render_agent_run(question: str, state: dict[str, Any], output_chars: int = 260) -> str:
    """Render one ``Agent.invoke`` result: the reasoning, each tool call and its result, then the answer."""
    lines = [f"### {question}", ""]
    for number, step in enumerate(state.get("tool_log", []), start=1):
        if step.get("thought"):
            lines.append(f"**Step {number} - reasoning:** {step['thought']}")
        try:
            arguments = json.dumps(json.loads(step["arguments"]), ensure_ascii=False)
        except (TypeError, ValueError):
            arguments = step["arguments"]
        lines.append(f"- 🔧 `{step['tool']}` {_shorten(arguments, 160)}")
        verdict = _judge_verdict(step) if step["tool"] == "evaluate_retrieval" else None
        if verdict:  # the judge's explanation is the agent's reasoning about the retrieval
            lines.append(f"  - evaluation: {verdict}")
        else:
            lines.append(f"  - result: {_shorten(step['output'], output_chars)}")
    if not state.get("tool_log"):
        lines.append("_No tools were called._")
    lines += ["", "**Final answer**", "", state["answer"]]
    return "\n".join(lines)
