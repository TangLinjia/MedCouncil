"""Run with python -m src.rag.cli build/query."""
import argparse
import json
from pathlib import Path

from .retriever import retrieve_guidelines
from .vectorstore import build_vectorstore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    query = commands.add_parser("query")
    query.add_argument("question")
    query.add_argument("--history", type=Path, help="JSON array of role/content turns")
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument("--mode", choices=["hybrid", "bm25", "dense", "rrf"], default="hybrid")
    args = parser.parse_args()
    if args.command == "build":
        store = build_vectorstore()
        print(json.dumps({"indexed_passages": len(store.get()["ids"])}))
    else:
        history = json.loads(args.history.read_text()) if args.history else []
        docs = retrieve_guidelines(args.question, args.top_k, history, mode=args.mode)
        print(json.dumps([{"content": d.page_content, **d.metadata} for d in docs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
