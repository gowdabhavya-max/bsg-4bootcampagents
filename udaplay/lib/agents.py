"""A tool-calling agent implemented as a state machine.

    prepare -> llm --(tool calls?)--> tools -> llm ... -> finish
"""
from __future__ import annotations

from typing import Sequence

from .llm import LLM
from .memory import ShortTermMemory
from .messages import AIMessage, Message, SystemMessage, ToolMessage, UserMessage
from .state_machine import END, State, StateMachine
from .tooling import Tool

MAX_ITERATIONS_ANSWER = (
    "I could not finish researching this question within the allowed number of steps. "
    "Please try rephrasing it or asking about something more specific."
)


class Agent:
    """Runs the model, executes the tools it asks for, and repeats until it answers."""

    def __init__(
        self,
        llm: LLM,
        instructions: str,
        tools: Sequence[Tool],
        memory: ShortTermMemory | None = None,
        max_iterations: int = 8,
    ):
        self.llm = llm
        self.instructions = instructions
        self.tools = {t.name: t for t in tools}
        self.memory = memory or ShortTermMemory()
        self.max_iterations = max_iterations
        self.machine = self._build_machine()

    # ---- state machine -------------------------------------------------
    def _build_machine(self) -> StateMachine:
        machine = StateMachine("agent", max_transitions=self.max_iterations * 2 + 4)
        machine.add_step("prepare", self._prepare, next="llm")
        machine.add_step("llm", self._call_llm, next=self._route_after_llm)
        machine.add_step("tools", self._run_tools, next="llm")
        machine.add_step("finish", self._finish, next=END)
        return machine

    def _prepare(self, state: State) -> State:
        history = self.memory.get(state["session_id"])
        messages: list[Message] = [SystemMessage(self.instructions), *history, UserMessage(state["query"])]
        return {"messages": messages, "iterations": 0, "tool_log": []}

    def _call_llm(self, state: State) -> State:
        ai = self.llm.invoke(state["messages"], tools=list(self.tools.values()))
        return {"messages": [*state["messages"], ai], "iterations": state["iterations"] + 1}

    def _route_after_llm(self, state: State) -> str:
        last: AIMessage = state["messages"][-1]
        if not last.tool_calls:
            return "finish"
        return "tools" if state["iterations"] < self.max_iterations else "finish"

    def _run_tools(self, state: State) -> State:
        messages = list(state["messages"])
        log = list(state["tool_log"])
        thought = messages[-1].content.strip()  # the model's stated reasoning for this step, if any
        for call in messages[-1].tool_calls:
            tool = self.tools.get(call.name)
            output = tool.run_from_json(call.arguments) if tool else f"Error: unknown tool '{call.name}'"
            log.append({"tool": call.name, "arguments": call.arguments, "output": output, "thought": thought})
            thought = ""  # parallel calls in one step share a single thought
            messages.append(ToolMessage(content=output, tool_call_id=call.id, name=call.name))
        return {"messages": messages, "tool_log": log}

    def _finish(self, state: State) -> State:
        last: AIMessage = state["messages"][-1]
        answer = last.content if not last.tool_calls and last.content else MAX_ITERATIONS_ANSWER
        # Persist only the visible exchange; tool chatter would bloat later prompts.
        self.memory.add([UserMessage(state["query"]), AIMessage(answer)], state["session_id"])
        return {"answer": answer}

    # ---- public API ----------------------------------------------------
    def invoke(self, query: str, session_id: str = "default") -> State:
        """Answer ``query``. The returned state has ``answer``, ``tool_log`` and ``trace``."""
        return self.machine.run({"query": query, "session_id": session_id})
