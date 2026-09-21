"""Exercise lib.LLM against the real OpenAI client, with HTTP mocked at the transport level."""
import json

import httpx
from openai import OpenAI

from lib import LLM, SystemMessage, UserMessage, tool
from schemas import EvaluationReport


def completion(message: dict) -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}],
    }


def make_llm(handler) -> tuple[LLM, list[dict]]:
    requests: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        status, payload = handler(body)
        return httpx.Response(status, json=payload)

    client = OpenAI(api_key="test", base_url="http://mock/v1", http_client=httpx.Client(transport=httpx.MockTransport(wrapped)))
    return LLM(model="test-model", client=client), requests


@tool
def retrieve_game(query: str) -> list:
    """Search.

    Args:
        query: text
    """
    return []


def test_invoke_sends_tools_and_parses_tool_calls():
    call = {"id": "call_1", "type": "function", "function": {"name": "retrieve_game", "arguments": '{"query": "FIFA 21"}'}}
    llm, requests = make_llm(lambda body: (200, completion({"content": None, "tool_calls": [call]})))

    ai = llm.invoke([SystemMessage("sys"), UserMessage("hi")], tools=[retrieve_game])

    assert ai.tool_calls[0].name == "retrieve_game" and ai.tool_calls[0].arguments == '{"query": "FIFA 21"}'
    body = requests[0]
    assert body["model"] == "test-model" and body["temperature"] == 0.0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["tools"][0]["function"]["name"] == "retrieve_game"


def test_parse_uses_structured_outputs():
    verdict = {"useful": True, "confidence": 0.9, "description": "ok", "missing_information": []}
    llm, requests = make_llm(lambda body: (200, completion({"content": json.dumps(verdict)})))

    report = llm.parse([UserMessage("judge")], EvaluationReport)

    assert isinstance(report, EvaluationReport) and report.useful and report.confidence == 0.9
    assert requests[0]["response_format"]["type"] == "json_schema"


def test_parse_falls_back_to_json_mode_when_endpoint_lacks_structured_outputs():
    verdict = {"useful": False, "confidence": 0.4, "description": "meh", "missing_information": ["x"]}

    def handler(body):
        if body["response_format"]["type"] == "json_schema":
            return 400, {"error": {"message": "response_format json_schema is not supported", "type": "invalid_request_error"}}
        return 200, completion({"content": json.dumps(verdict)})

    llm, requests = make_llm(handler)
    report = llm.parse([UserMessage("judge")], EvaluationReport)

    assert report.useful is False and report.missing_information == ["x"]
    assert [r["response_format"]["type"] for r in requests] == ["json_schema", "json_object"]
    assert "JSON schema" in requests[1]["messages"][0]["content"]
