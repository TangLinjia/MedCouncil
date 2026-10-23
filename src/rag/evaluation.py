"""Retrieval evaluation with explicit annotation provenance."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .retriever import retrieve_guidelines, rewrite_query
from .chunker import corpus_fingerprint
from langchain_core.documents import Document


def validate_corpus(cases: list[dict], documents: list[Document]) -> None:
    fingerprint = corpus_fingerprint(documents)
    for case in cases:
        if case.get("corpus_sha256") != fingerprint:
            raise ValueError(f"Stale corpus labels for {case['id']}; review against the current corpus before evaluation.")


def retrieval_metrics(retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
    if k < 1 or not relevant:
        raise ValueError("k and relevance labels must be nonempty.")
    seen = set()
    hits = []
    for passage_id in retrieved[:k]:
        hits.append(int(passage_id in relevant and passage_id not in seen))
        seen.add(passage_id)
    recall = sum(hits) / len(relevant)
    mrr = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {"recall": recall, "hit_rate": float(any(hits)), "mrr": mrr, "ndcg": dcg / ideal}


def select_queries(cases: list[dict], include_unreviewed: bool) -> list[dict]:
    for case in cases:
        if case["annotation_status"] == "human_reviewed" and not (case.get("reviewer") or "").strip():
            raise ValueError("Human-reviewed labels must name a reviewer.")
    return [case for case in cases if include_unreviewed or case["annotation_status"] == "human_reviewed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/dev.json"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/results.json"))
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--ablations", action="store_true", help="Compare BM25, dense, RRF and hybrid")
    args = parser.parse_args()
    if args.top_k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("top-k and limit must be positive")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("split") == "test":
        parser.error("Use python -m src.rag.benchmark test for frozen held-out evaluation.")
    cases = select_queries(dataset["queries"], args.include_unreviewed)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        parser.error("No human-reviewed queries. Review labels or explicitly pass --include-unreviewed for a synthetic smoke evaluation.")
    modes = ["bm25", "dense", "rrf", "hybrid"] if args.ablations else ["hybrid"]
    # Fail rather than silently score stale/mismatched gold labels as retrieval errors.
    from .vectorstore import get_vectorstore
    indexed = get_vectorstore().get(include=["documents", "metadatas"])
    indexed_ids = set(indexed["ids"])
    validate_corpus(cases, [Document(page_content=text, metadata=meta)
                           for text, meta in zip(indexed["documents"], indexed["metadatas"])])
    for case in cases:
        if not case["relevant_passage_ids"] or not set(case["relevant_passage_ids"]) <= indexed_ids:
            raise ValueError(f"Unknown or empty gold passage IDs for {case['id']}; rebuild and review labels.")
    results = []
    for case in cases:
        start = time.perf_counter()
        rewritten = rewrite_query(case["query"], case["history"])
        row = {"id": case["id"], "query": case["query"], "rewritten_query": rewritten,
               "annotation_status": case["annotation_status"], "modes": {}}
        for mode in modes:
            docs = retrieve_guidelines(rewritten, args.top_k, mode=mode)
            ids = [d.metadata["passage_id"] for d in docs]
            row["modes"][mode] = {"retrieved_ids": ids, **retrieval_metrics(ids, set(case["relevant_passage_ids"]), args.top_k)}
        row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        results.append(row)
        print(f"{case['id']} {len(results)}/{len(cases)}", flush=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "top_k": args.top_k,
              "relevance_unit": "passage", "aggregation": "macro average over queries",
              "recall_definition": "number of distinct relevant passages in top k / number of all labeled relevant passages",
              "hit_rate_definition": "fraction of queries with at least one relevant passage in top k",
              "dataset": str(args.dataset), "scope": dataset["scope"], "total": len(results),
              "annotation_counts": dict(Counter(row["annotation_status"] for row in cases)),
              "summary": {mode: {metric: sum(row["modes"][mode][metric] for row in results) / len(results)
                                  for metric in ("recall", "hit_rate", "mrr", "ndcg")} for mode in modes},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
