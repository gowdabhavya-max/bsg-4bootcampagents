"""Command-line entry point.

    python main.py "Who developed FIFA 21?"          # structured workflow (default)
    python main.py --mode agent "When was Pokémon Gold and Silver released?"
    python main.py --reload                          # rebuild the games vector DB
    python main.py                                   # interactive chat
"""
from __future__ import annotations

import argparse

from app import create_app
from report import render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="UdaPlay - video game research agent")
    parser.add_argument("question", nargs="*", help="question to ask; omit for interactive mode")
    parser.add_argument("--mode", choices=["workflow", "agent"], default="workflow",
                        help="workflow: state machine with structured report; agent: tool-calling agent")
    parser.add_argument("--reload", action="store_true", help="re-embed the games dataset before answering")
    parser.add_argument("--trace", action="store_true", help="show the workflow steps that ran")
    args = parser.parse_args()

    try:
        app = create_app()
    except RuntimeError as exc:  # missing API keys
        raise SystemExit(f"error: {exc}")
    if args.reload:
        print(f"Loaded {app.games.load_games(app.settings.games_dir)} games into the vector DB.")

    def answer(question: str) -> None:
        if args.mode == "agent":
            print(app.agent.invoke(question)["answer"])
        else:
            print(render_markdown(app.workflow.ask(question), show_trace=args.trace))

    if args.question:
        answer(" ".join(args.question))
        return
    print("UdaPlay - ask about video games (blank line to quit).")
    while (question := input("\n> ").strip()):
        answer(question)


if __name__ == "__main__":
    main()
