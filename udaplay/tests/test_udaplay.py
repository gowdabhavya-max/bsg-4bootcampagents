import json
from pathlib import Path

import pytest

from conftest import FakeLLM, FakeTavily
from long_term_memory import LongTermMemory
from report import render_markdown
from schemas import DraftAnswer, EvaluationReport, ExtractedMemories, MemoryEntry
from tools import build_tools
from workflow import UdaPlay

GAMES_DIR = Path(__file__).resolve().parents[1] / "games"

WEB_RESULT = {
    "title": "Rockstar Games - news",
    "url": "https://example.com/rockstar",
    "content": "Rockstar Games is developing Grand Theft Auto VI.",
    "score": 0.9,
}


# ---- dataset ---------------------------------------------------------------
def test_dataset_is_well_formed():
    files = sorted(GAMES_DIR.glob("*.json"))
    assert len(files) >= 30
    required = {"Name", "Platform", "Genre", "Publisher", "Developer", "Description", "YearOfRelease"}
    seen = set()
    for f in files:
        game = json.loads(f.read_text(encoding="utf-8"))
        assert required <= game.keys(), f.name
        assert isinstance(game["YearOfRelease"], int)
        seen.add((game["Name"], game["Platform"]))
    assert len(seen) == len(files), "duplicate (game, platform) records"


# ---- vector store ------------------------------------------------------------
def test_load_games_is_idempotent(stores):
    games, _ = stores
    n = games.count()
    assert n == len(list(GAMES_DIR.glob("*.json")))
    games.load_games(GAMES_DIR)
    assert games.count() == n


@pytest.mark.parametrize(
    "query, expected_name",
    [
        ("Who developed FIFA 21?", "FIFA 21"),
        ("When was God of War Ragnarok released?", "God of War Ragnarök"),
        ("What platform was Pokémon Red launched on?", "Pokémon Red"),
    ],
)
def test_semantic_search_finds_the_right_game(stores, query, expected_name):
    games, _ = stores
    hits = games.query(query, n_results=3)
    assert expected_name in [h["metadata"]["Name"] for h in hits]


def test_query_on_empty_store_returns_nothing(stores):
    _, memory = stores
    assert memory.query("anything") == []


def test_reset_empties_the_collection(stores):
    _, memory = stores
    LongTermMemory(memory).remember([MemoryEntry(topic="t", fact="f", source_url="u")])
    assert memory.count() == 1
    memory.reset()
    assert memory.count() == 0


# ---- tools ------------------------------------------------------------------
def make_tools(stores, llm=None, tavily=None):
    games, memory = stores
    tools = build_tools(games, memory, llm or FakeLLM(), tavily or FakeTavily(results=[WEB_RESULT]))
    return {t.name: t for t in tools}


def test_three_tools_with_openai_schemas(stores):
    tools = make_tools(stores)
    assert set(tools) == {"retrieve_game", "evaluate_retrieval", "game_web_search"}
    assert tools["retrieve_game"].schema["function"]["parameters"]["required"] == ["query"]
    assert tools["evaluate_retrieval"].schema["function"]["parameters"]["required"] == ["question", "retrieved_docs"]
    assert "query" in tools["retrieve_game"].schema["function"]["parameters"]["properties"]


def test_retrieve_game_merges_internal_and_memory(stores):
    _, memory = stores
    LongTermMemory(memory).remember(
        [MemoryEntry(topic="Rockstar Games", fact="Rockstar Games is developing Grand Theft Auto VI.",
                     source_url="https://example.com/rockstar")]
    )
    hits = make_tools(stores)["retrieve_game"](query="What is Rockstar Games working on?")
    assert {h["source"] for h in hits} == {"internal", "memory"}
    assert [h["distance"] for h in hits] == sorted(h["distance"] for h in hits)
    memory_hit = next(h for h in hits if h["source"] == "memory")
    assert memory_hit["source_url"] == "https://example.com/rockstar"


def test_evaluate_retrieval_short_circuits_on_no_documents(stores):
    llm = FakeLLM()
    result = make_tools(stores, llm)["evaluate_retrieval"](question="q", retrieved_docs=[])
    assert result["useful"] is False and llm.parse_calls == []


def test_evaluate_retrieval_uses_judge_and_clamps_confidence(stores):
    verdict = EvaluationReport(useful=True, confidence=1.7, description="direct hit")
    llm = FakeLLM(parsed={EvaluationReport: verdict})
    docs = [{"Name": "FIFA 21", "Developer": "EA Vancouver"}]
    result = make_tools(stores, llm)["evaluate_retrieval"](question="Who developed FIFA 21?", retrieved_docs=json.dumps(docs))
    assert result["useful"] is True and result["confidence"] == 1.0
    _, messages = llm.parse_calls[0]
    assert "EA Vancouver" in messages[1].content  # string-encoded docs were decoded and passed on


def test_web_search_tool_trims_and_shapes_results(stores):
    tavily = FakeTavily(results=[{**WEB_RESULT, "content": "x" * 5000}], answer="GTA VI")
    out = make_tools(stores, tavily=tavily)["game_web_search"](question="q")
    assert out["summary"] == "GTA VI"
    assert len(out["results"][0]["content"]) == 900
    assert tavily.queries == ["q"]


# ---- workflow ---------------------------------------------------------------
def build_workflow(stores, llm, tavily, threshold=0.6):
    games, memory = stores
    tools = build_tools(games, memory, llm, tavily)
    return UdaPlay(llm, tools, memory=LongTermMemory(memory), confidence_threshold=threshold)


def test_internal_answer_skips_web_search(stores):
    llm = FakeLLM(parsed={
        EvaluationReport: EvaluationReport(useful=True, confidence=0.95, description="FIFA 21 record found"),
        DraftAnswer: lambda msgs: DraftAnswer(
            answer="FIFA 21 was developed by EA Vancouver and EA Romania [D1].", confidence="high", evidence_ids=["D1"]),
    })
    tavily = FakeTavily()
    report = build_workflow(stores, llm, tavily).ask("Who developed FIFA 21?")

    assert report.trace == ["retrieve", "evaluate", "answer"]
    assert tavily.queries == [] and report.used_web_search is False
    assert report.confidence == "high"
    assert report.sources[0].kind == "internal" and report.sources[0].reference.startswith("game record")
    assert "Web search used:** no" in render_markdown(report)


def test_low_quality_retrieval_falls_back_to_web_and_remembers(stores):
    games, memory = stores
    llm = FakeLLM(parsed={
        EvaluationReport: EvaluationReport(useful=False, confidence=0.9, description="static data cannot say"),
        ExtractedMemories: ExtractedMemories(entries=[
            MemoryEntry(topic="Rockstar Games", fact="As of today Rockstar Games is developing Grand Theft Auto VI.",
                        entities=["Rockstar Games", "Grand Theft Auto VI"], source_url=WEB_RESULT["url"])]),
        DraftAnswer: DraftAnswer(answer="Rockstar is working on GTA VI [W1].", confidence="medium",
                                 evidence_ids=["W1", "W1", "W99"]),  # duplicate + invented id
    })
    tavily = FakeTavily(results=[WEB_RESULT])
    report = build_workflow(stores, llm, tavily).ask("What is Rockstar Games working on right now?")

    assert report.trace == ["retrieve", "evaluate", "web_search", "remember", "answer"]
    assert report.used_web_search and report.saved_to_memory == 1
    assert [(s.kind, s.reference) for s in report.sources] == [("web", WEB_RESULT["url"])]
    assert memory.count() == 1


def test_confident_but_not_useful_and_useful_but_unsure_both_trigger_web(stores):
    for useful, confidence in [(False, 0.99), (True, 0.3)]:
        llm = FakeLLM(parsed={
            EvaluationReport: EvaluationReport(useful=useful, confidence=confidence, description="d"),
            ExtractedMemories: ExtractedMemories(),
            DraftAnswer: DraftAnswer(answer="a", confidence="low", evidence_ids=[]),
        })
        report = build_workflow(stores, llm, FakeTavily(results=[WEB_RESULT])).ask("q")
        assert "web_search" in report.trace, (useful, confidence)


def test_learned_fact_is_reused_without_a_second_web_search(stores):
    """After learning from the web once, the same question is answered from long-term memory."""
    verdicts = iter([
        EvaluationReport(useful=False, confidence=0.9, description="need web"),
        EvaluationReport(useful=True, confidence=0.9, description="memory has it"),
    ])
    llm = FakeLLM(parsed={
        EvaluationReport: lambda msgs: next(verdicts),
        ExtractedMemories: ExtractedMemories(entries=[
            MemoryEntry(topic="Rockstar Games", fact="Rockstar Games is developing Grand Theft Auto VI.",
                        source_url=WEB_RESULT["url"])]),
        DraftAnswer: DraftAnswer(answer="GTA VI.", confidence="medium", evidence_ids=["M1"]),
    })
    tavily = FakeTavily(results=[WEB_RESULT])
    workflow = build_workflow(stores, llm, tavily)
    workflow.ask("What is Rockstar Games working on?")
    second = workflow.ask("What is Rockstar Games working on?")

    assert len(tavily.queries) == 1
    assert second.trace == ["retrieve", "evaluate", "answer"]
    assert [s.kind for s in second.sources] == ["memory"]
    # the model was shown the remembered fact as evidence
    _, messages = llm.parse_calls[-1]
    assert "developing Grand Theft Auto VI" in messages[1].content


def test_web_failure_degrades_gracefully(stores):
    llm = FakeLLM(parsed={
        EvaluationReport: EvaluationReport(useful=False, confidence=0.9, description="d"),
        DraftAnswer: DraftAnswer(answer="Partial answer from records.", confidence="high", evidence_ids=["D1"]),
    })
    report = build_workflow(stores, llm, FakeTavily(error=TimeoutError("boom"))).ask("q")
    assert report.trace == ["retrieve", "evaluate", "web_search", "remember", "answer"]
    assert report.used_web_search is False
    assert report.confidence == "medium"  # a "high" claim is capped when the web check failed
    assert ExtractedMemories not in [schema for schema, _ in llm.parse_calls]


def test_memory_write_failure_does_not_break_the_answer(stores):
    class Boom(FakeLLM):
        def parse(self, messages, schema):
            if schema is ExtractedMemories:
                raise RuntimeError("extractor down")
            return super().parse(messages, schema)

    llm = Boom(parsed={
        EvaluationReport: EvaluationReport(useful=False, confidence=0.9, description="d"),
        DraftAnswer: DraftAnswer(answer="From the web [W1].", confidence="medium", evidence_ids=["W1"]),
    })
    report = build_workflow(stores, llm, FakeTavily(results=[WEB_RESULT])).ask("q")
    assert report.saved_to_memory == 0 and report.sources[0].kind == "web"


def test_no_evidence_at_all_returns_low_confidence_without_calling_the_model(stores):
    games, memory = stores
    games.reset()  # empty knowledge base
    llm = FakeLLM()
    report = build_workflow((games, memory), llm, FakeTavily(results=[])).ask("anything")
    assert report.confidence == "low" and "couldn't find" in report.answer
    assert [schema for schema, _ in llm.parse_calls] == []  # judge short-circuits, no answer call


def test_render_markdown_includes_sources_and_trace(stores):
    llm = FakeLLM(parsed={
        EvaluationReport: EvaluationReport(useful=True, confidence=0.9, description="ok"),
        DraftAnswer: DraftAnswer(answer="ans", confidence="high", evidence_ids=["D1"]),
    })
    report = build_workflow(stores, llm, FakeTavily()).ask("Who developed FIFA 21?")
    text = render_markdown(report, show_trace=True)
    assert "**Sources**" in text and "retrieve → evaluate → answer" in text
