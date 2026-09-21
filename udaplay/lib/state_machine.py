"""A minimal state machine: named steps that update a shared state dict, joined by
fixed or conditional transitions."""
from __future__ import annotations

from typing import Any, Callable, Union

END = "__end__"

State = dict[str, Any]
Step = Callable[[State], Union[State, None]]
Router = Callable[[State], str]


class StateMachineError(RuntimeError):
    pass


class StateMachine:
    """Run ``step`` functions until one transitions to :data:`END`.

    * A step receives the state and returns a dict of updates (or ``None``).
    * The transition after a step is either a step name / ``END``, or a router
      function ``state -> name`` for conditional branching.
    * ``state["trace"]`` records every step that ran, in order.
    """

    def __init__(self, name: str = "state_machine", max_transitions: int = 25):
        self.name = name
        self.max_transitions = max_transitions
        self._steps: dict[str, Step] = {}
        self._next: dict[str, Union[str, Router]] = {}
        self._entry: str | None = None

    def add_step(self, name: str, fn: Step, next: Union[str, Router] = END) -> "StateMachine":
        if name in self._steps:
            raise StateMachineError(f"Step '{name}' already exists")
        self._steps[name] = fn
        self._next[name] = next
        if self._entry is None:
            self._entry = name
        return self

    def set_entry(self, name: str) -> "StateMachine":
        self._entry = name
        return self

    def _validate(self) -> None:
        if self._entry not in self._steps:
            raise StateMachineError("No entry step defined")
        for src, nxt in self._next.items():
            if isinstance(nxt, str) and nxt != END and nxt not in self._steps:
                raise StateMachineError(f"Step '{src}' transitions to unknown step '{nxt}'")

    def run(self, state: State | None = None) -> State:
        self._validate()
        state = dict(state or {})
        state["trace"] = []
        current = self._entry
        for _ in range(self.max_transitions):
            state["trace"].append(current)
            state.update(self._steps[current](state) or {})
            nxt = self._next[current]
            current = nxt(state) if callable(nxt) else nxt
            if current == END:
                return state
            if current not in self._steps:
                raise StateMachineError(f"Router after '{state['trace'][-1]}' returned unknown step '{current}'")
        raise StateMachineError(f"'{self.name}' exceeded {self.max_transitions} transitions (possible loop)")
