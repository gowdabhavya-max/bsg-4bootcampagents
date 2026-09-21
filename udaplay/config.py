"""Environment loading and shared factory helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://openai.vocareum.com/v1"


def load_env() -> None:
    """Load ``.env`` (and ``config.env``, as used in the course workspace) if present."""
    for name in (".env", "config.env"):
        for directory in (Path.cwd(), PROJECT_DIR):
            candidate = directory / name
            if candidate.is_file():
                load_dotenv(candidate, override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    tavily_api_key: str
    openai_base_url: str
    chat_model: str
    embedding_model: str
    chroma_path: Path
    games_dir: Path

    @classmethod
    def from_env(cls, require_keys: bool = True) -> "Settings":
        load_env()
        settings = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            chat_model=os.getenv("UDAPLAY_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("UDAPLAY_EMBEDDING_MODEL", "text-embedding-3-small"),
            chroma_path=Path(os.getenv("UDAPLAY_CHROMA_PATH", PROJECT_DIR / "chromadb")),
            games_dir=Path(os.getenv("UDAPLAY_GAMES_DIR", PROJECT_DIR / "games")),
        )
        if require_keys:
            missing = [k for k, v in (("OPENAI_API_KEY", settings.openai_api_key),
                                      ("TAVILY_API_KEY", settings.tavily_api_key)) if not v]
            if missing:
                raise RuntimeError(
                    f"Missing {', '.join(missing)}. Copy .env.example to .env and fill in your keys."
                )
        return settings
