"""A tiny ``@tool`` decorator that turns a typed, documented function into an
OpenAI function-calling tool."""
from __future__ import annotations

import inspect
import json
import re
import typing
from dataclasses import dataclass
from typing import Any, Callable

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_ARG_LINE = re.compile(r"^\s*(?:[-*]\s*)?(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = typing.get_origin(annotation)
    if origin in (list, tuple, set):
        return {"type": "array", "items": {}}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if origin is typing.Union:  # Optional[X] -> X
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if args else {"type": "string"}
    if annotation in (list, tuple, set):
        return {"type": "array", "items": {}}
    return {"type": _JSON_TYPES.get(annotation, "string")}


def _split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return (description, {arg: description}) from a docstring whose argument
    section starts with a line reading ``Args:`` / ``args:`` / ``Arguments:``."""
    lines = inspect.cleandoc(doc or "").splitlines()
    description: list[str] = []
    args: dict[str, str] = {}
    section = "description"
    current: str | None = None
    for line in lines:
        header = line.strip().lower().rstrip(":")
        if header in {"args", "arguments", "parameters"}:
            section, current = "args", None
            continue
        if header in {"returns", "return"}:
            section, current = "returns", None
            continue
        if section == "description":
            description.append(line)
        elif section == "args":
            match = _ARG_LINE.match(line)
            if match:
                current = match.group(1)
                args[current] = match.group(2).strip()
            elif current and line.strip():  # wrapped continuation line
                args[current] += " " + line.strip()
    return " ".join(part.strip() for part in description if part.strip()), args


@dataclass
class Tool:
    """A callable the LLM may invoke."""

    fn: Callable[..., Any]
    name: str
    description: str
    parameters: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run_from_json(self, raw_arguments: str) -> str:
        """Execute with model-supplied JSON arguments and return a string result.

        Errors are returned as text (not raised) so the model can see them and recover.
        """
        try:
            kwargs = json.loads(raw_arguments or "{}")
            result = self.fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model on purpose
            return f"Error running tool '{self.name}': {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


def tool(fn: Callable[..., Any]) -> Tool:
    """Decorator: build the JSON schema from type hints and the docstring."""
    description, arg_docs = _split_docstring(fn.__doc__ or "")
    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        prop = _json_type(hints.get(name, str))
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    parameters = {"type": "object", "properties": properties, "required": required}
    return Tool(fn=fn, name=fn.__name__, description=description, parameters=parameters)
