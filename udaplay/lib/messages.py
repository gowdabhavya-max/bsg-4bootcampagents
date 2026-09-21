"""Chat message types that serialise to the OpenAI chat-completions format."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    content: str = ""

    role = "user"  # overridden by subclasses

    def to_openai(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class SystemMessage(Message):
    role = "system"


@dataclass
class UserMessage(Message):
    role = "user"


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class AIMessage(Message):
    tool_calls: list[ToolCall] = field(default_factory=list)

    role = "assistant"

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content or None}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        return payload


@dataclass
class ToolMessage(Message):
    tool_call_id: str = ""
    name: str = ""

    role = "tool"

    def to_openai(self) -> dict[str, Any]:
        return {"role": self.role, "tool_call_id": self.tool_call_id, "content": self.content}
