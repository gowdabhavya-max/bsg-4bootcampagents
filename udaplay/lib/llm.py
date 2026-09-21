"""Thin wrapper around an OpenAI-compatible chat-completions endpoint."""
from __future__ import annotations

import json
import os
from typing import Sequence, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .messages import AIMessage, Message, SystemMessage, ToolCall
from .tooling import Tool

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gpt-4o-mini"


class LLM:
    """Chat model with tool calling and schema-validated (structured) output."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
    ):
        self.model = model or os.getenv("UDAPLAY_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.client = client or OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
        )

    def invoke(self, messages: Sequence[Message], tools: Sequence[Tool] | None = None) -> AIMessage:
        """One chat turn. Returns an AIMessage that may carry tool calls."""
        kwargs: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [m.to_openai() for m in messages],
        }
        if tools:
            kwargs["tools"] = [t.schema for t in tools]
        message = self.client.chat.completions.create(**kwargs).choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
            for c in (message.tool_calls or [])
        ]
        return AIMessage(content=message.content or "", tool_calls=calls)

    def parse(self, messages: Sequence[Message], schema: type[T]) -> T:
        """Ask for output matching a pydantic ``schema`` and return the validated object.

        Uses native structured outputs when the endpoint supports them and falls back
        to JSON mode (schema embedded in the prompt) otherwise.
        """
        payload = [m.to_openai() for m in messages]
        try:
            completions = getattr(self.client.chat.completions, "parse", None) or self.client.beta.chat.completions.parse
            completion = completions(
                model=self.model, temperature=self.temperature, messages=payload, response_format=schema
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception:  # noqa: BLE001 - endpoint without structured outputs -> JSON mode below
            pass

        instruction = SystemMessage(
            "Reply with a single JSON object that validates against this JSON schema, and nothing else:\n"
            + json.dumps(schema.model_json_schema())
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[instruction.to_openai(), *payload],
            response_format={"type": "json_object"},
        )
        return schema.model_validate_json(completion.choices[0].message.content or "{}")
