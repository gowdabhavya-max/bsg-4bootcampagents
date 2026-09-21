import pytest

from lib import Agent, AIMessage, END, ShortTermMemory, StateMachine, ToolCall, tool
from lib.state_machine import StateMachineError

from conftest import FakeLLM


# ---- @tool ---------------------------------------------------------------
def test_tool_schema_from_signature_and_docstring():
    @tool
    def lookup(query: str, limit: int = 3, docs: list = None) -> str:
        """Find things.

        Args:
            query: what to look for
            limit: maximum number of hits,
                wrapped onto a second line
            docs: prior documents
        """
        return f"{query}:{limit}"

    fn = lookup.schema["function"]
    assert fn["name"] == "lookup"
    assert fn["description"] == "Find things."
    props = fn["parameters"]["properties"]
    assert props["query"] == {"type": "string", "description": "what to look for"}
    assert props["limit"]["type"] == "integer"
    assert props["limit"]["description"] == "maximum number of hits, wrapped onto a second line"
    assert props["docs"]["type"] == "array"
    assert fn["parameters"]["required"] == ["query"]


def test_tool_run_from_json_serialises_and_reports_errors():
    @tool
    def add(a: int, b: int) -> dict:
        """Add.

        Args:
            a: first
            b: second
        """
        return {"sum": a + b}

    assert add.run_from_json('{"a": 1, "b": 2}') == '{"sum": 3}'
    assert add.run_from_json('{"a": 1}').startswith("Error running tool 'add'")
    assert add.run_from_json("not json").startswith("Error running tool 'add'")


# ---- StateMachine -----------------------------------------------------------
def test_state_machine_branches_and_traces():
    m = StateMachine("t")
    m.add_step("start", lambda s: {"n": s["n"] + 1}, next=lambda s: "big" if s["n"] > 5 else "small")
    m.add_step("small", lambda s: {"label": "small"})
    m.add_step("big", lambda s: {"label": "big"})
    assert m.run({"n": 1})["label"] == "small"
    out = m.run({"n": 9})
    assert out["label"] == "big" and out["trace"] == ["start", "big"]


def test_state_machine_detects_loops_and_bad_transitions():
    loop = StateMachine("loop", max_transitions=5)
    loop.add_step("a", lambda s: None, next="a")
    with pytest.raises(StateMachineError, match="exceeded"):
        loop.run()

    bad = StateMachine("bad")
    bad.add_step("a", lambda s: None, next="nowhere")
    with pytest.raises(StateMachineError, match="unknown step"):
        bad.run()

    routed = StateMachine("routed")
    routed.add_step("a", lambda s: None, next=lambda s: "ghost")
    with pytest.raises(StateMachineError, match="unknown step"):
        routed.run()


# ---- Agent ---------------------------------------------------------------
@tool
def echo(text: str) -> str:
    """Echo text back.

    Args:
        text: text to echo
    """
    return f"echo:{text}"


def test_agent_runs_tools_then_answers_and_remembers_the_session():
    call = ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')
    llm = FakeLLM(chat_script=[
        AIMessage("I should echo it.", tool_calls=[call]), AIMessage("done: echo:hi"), AIMessage("second"),
    ])
    agent = Agent(llm, "be brief", [echo], memory=ShortTermMemory())

    state = agent.invoke("say hi", session_id="s1")
    assert state["answer"] == "done: echo:hi"
    assert state["tool_log"] == [
        {"tool": "echo", "arguments": '{"text": "hi"}', "output": "echo:hi", "thought": "I should echo it."}
    ]
    assert state["trace"] == ["prepare", "llm", "tools", "llm", "finish"]

    agent.invoke("and again?", session_id="s1")
    second_prompt = llm.invocations[-1]
    assert [m.content for m in second_prompt[1:]] == ["say hi", "done: echo:hi", "and again?"]

    agent.invoke("fresh", session_id="other")
    assert len(llm.invocations[-1]) == 2  # system + user only: sessions are isolated


def test_agent_reports_unknown_tools_to_the_model():
    call = ToolCall(id="c1", name="missing", arguments="{}")
    llm = FakeLLM(chat_script=[AIMessage("", tool_calls=[call]), AIMessage("sorry")])
    state = Agent(llm, "x", [echo]).invoke("q")
    assert "unknown tool" in state["tool_log"][0]["output"]
    assert state["answer"] == "sorry"


def test_agent_stops_after_max_iterations():
    call = ToolCall(id="c", name="echo", arguments='{"text": "again"}')
    llm = FakeLLM(chat_script=[AIMessage("", tool_calls=[call]) for _ in range(10)])
    agent = Agent(llm, "x", [echo], max_iterations=3)
    state = agent.invoke("loop forever")
    assert len(state["tool_log"]) == 2  # 3rd model reply hits the cap before its tools run
    assert "could not finish" in state["answer"]
    # the cap must not leave an assistant tool call without tool output in persisted memory
    assert [m.content for m in agent.memory.get()] == ["loop forever", state["answer"]]


def test_render_agent_run_shows_reasoning_tools_and_answer():
    from report import render_agent_run

    call = ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')
    llm = FakeLLM(chat_script=[AIMessage("Echoing first.", tool_calls=[call]), AIMessage("done: echo:hi")])
    state = Agent(llm, "x", [echo]).invoke("say hi")
    text = render_agent_run("say hi", state)
    assert "reasoning:** Echoing first." in text
    assert '`echo` {"text": "hi"}' in text and "result: echo:hi" in text
    assert text.rstrip().endswith("done: echo:hi")


def test_render_agent_run_summarises_the_judge_verdict():
    from report import render_agent_run

    state = {
        "answer": "1999",
        "tool_log": [{
            "tool": "evaluate_retrieval", "arguments": "{}", "thought": "",
            "output": '{"useful": false, "confidence": 0.9, "description": "Only PS4 is mentioned."}',
        }],
    }
    text = render_agent_run("q", state)
    assert "evaluation: NOT useful (confidence 90%) - Only PS4 is mentioned." in text
