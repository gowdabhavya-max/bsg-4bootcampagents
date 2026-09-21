"""Short-term (per-session) conversation memory."""
from __future__ import annotations

from collections import defaultdict

from .messages import Message


class ShortTermMemory:
    """Keeps the recent conversation for each session id."""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._sessions: dict[str, list[Message]] = defaultdict(list)

    def get(self, session_id: str = "default") -> list[Message]:
        return list(self._sessions[session_id])

    def add(self, messages: list[Message], session_id: str = "default") -> None:
        history = self._sessions[session_id]
        history.extend(messages)
        del history[: max(0, len(history) - self.max_messages)]

    def clear(self, session_id: str = "default") -> None:
        self._sessions.pop(session_id, None)
