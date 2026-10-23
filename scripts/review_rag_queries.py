"""Interactive, per-query HUMAN relevance review; never auto-approve a dataset."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.rag.chunker import split_documents, corpus_fingerprint
from src.rag.loader import load_guidelines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/dev.json"),
                        help="Review dev.json and test.json separately before formal evaluation")
    args = parser.parse_args()
    if not args.reviewer.strip():
        parser.error("A real reviewer name is required")
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    documents = split_documents(load_guidelines())
    fingerprint = corpus_fingerprint(documents)
    passages = {d.metadata["passage_id"]: d.page_content for d in documents}
    for row in data["queries"]:
        if row["annotation_status"] == "human_reviewed" and row.get("corpus_sha256") == fingerprint:
            continue
        print(json.dumps({key: row[key] for key in ("id", "history", "query", "relevant_passage_ids", "rationale")},
                         ensure_ascii=False, indent=2))
        for passage_id, text in passages.items():
            print(f"\n[{passage_id}]\n{text}")
        answer = input("Enter=accept candidate labels; comma-separated IDs=correct labels; s=skip; q=quit: ").strip()
        if answer == "q":
            break
        if answer == "s":
            continue
        labels = [part.strip() for part in answer.split(",")] if answer else row["relevant_passage_ids"]
        if not labels or not set(labels) <= passages.keys():
            print("Unknown/empty passage IDs: skipped, no annotation saved.")
            continue
        row.update(relevant_passage_ids=list(dict.fromkeys(labels)), annotation_status="human_reviewed",
                   reviewer=args.reviewer.strip(), reviewed_at=datetime.now(timezone.utc).isoformat(),
                   corpus_sha256=fingerprint)
        if answer:
            row["rationale"] = input("Reason for corrected labels: ").strip()
        temporary = args.dataset.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.dataset)
        print("Saved", row["id"])


if __name__ == "__main__":
    main()
