"""Wire everything together: stores, tools, the tool-calling agent and the workflow."""
from __future__ import annotations

from dataclasses import dataclass

from lib import Agent, LLM, ShortTermMemory, Tool
from config import Settings
from long_term_memory import LongTermMemory
from tools import build_tools
from vector_store import GAMES_COLLECTION, MEMORY_COLLECTION, VectorStoreManager, make_embedding_function
from workflow import UdaPlay

AGENT_INSTRUCTIONS = """\
You are UdaPlay, an AI research agent for the video game industry. Answer questions about games: \
titles, release dates, platforms, genres, developers, publishers and company news.

Before every tool call, write one short sentence saying what you are about to do and why.

Follow this process for every question:
1. Call retrieve_game with the question to search the internal knowledge base.
2. Call evaluate_retrieval with the question and the retrieved documents.
3. If the evaluation says the documents are useful, answer from them.
   Otherwise call game_web_search and answer from the web results (combine with the internal \
documents when both help).
4. Answer only from tool results, never from memory. If the evidence does not answer the question \
(or contradicts its premise), say so plainly.

Format: a short natural-language answer, then a "Sources" list naming where each fact came from \
(internal game record, remembered fact, or the web URL). Mention your confidence (high / medium / low).
"""


@dataclass
class UdaPlayApp:
    settings: Settings
    llm: LLM
    games: VectorStoreManager
    memory_store: VectorStoreManager
    tools: list[Tool]
    agent: Agent
    workflow: UdaPlay


def create_app(settings: Settings | None = None, load_games: bool = True, tavily_client=None) -> UdaPlayApp:
    """Build the whole system. Loads the games dataset into Chroma on first use."""
    settings = settings or Settings.from_env()
    embedding_fn = make_embedding_function(settings)
    client = VectorStoreManager.persistent_client(settings.chroma_path)
    games = VectorStoreManager(client, GAMES_COLLECTION, embedding_fn)
    memory_store = VectorStoreManager(client, MEMORY_COLLECTION, embedding_fn)
    if load_games and games.count() == 0:
        games.load_games(settings.games_dir)

    if tavily_client is None:
        from tavily import TavilyClient

        tavily_client = TavilyClient(api_key=settings.tavily_api_key)

    llm = LLM(model=settings.chat_model, api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    tools = build_tools(games, memory_store, llm, tavily_client)
    agent = Agent(llm, AGENT_INSTRUCTIONS, tools, memory=ShortTermMemory())
    workflow = UdaPlay(llm, tools, memory=LongTermMemory(memory_store))
    return UdaPlayApp(settings, llm, games, memory_store, tools, agent, workflow)
