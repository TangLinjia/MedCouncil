import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.rag import retriever, vectorstore
from src.rag.chunker import split_documents
from test.test_storage import FixedEmbeddings


def doc(text, passage_id):
    return Document(page_content=text, metadata={"passage_id": passage_id})


def test_sections_have_stable_unique_ids():
    source = Document(page_content="# Pneumonia\n\n## Diagnosis\nCough.\n\n## Treatment\nAllergies.",
                      metadata={"source": "pneumonia.md"})
    chunks = split_documents([source])
    assert [d.metadata["passage_id"] for d in chunks] == [
        "pneumonia.md#diagnosis-0", "pneumonia.md#treatment-0"]
    assert chunks == split_documents([source])


def test_rrf_rewards_agreement_without_duplicates():
    a, b, c = doc("a", "a"), doc("b", "b"), doc("c", "c")
    result = retriever.reciprocal_rank_fusion([[a, b, b], [c, b]])
    assert [d.metadata["passage_id"] for d in result] == ["b", "a", "c"]


def test_chinese_bm25_and_unmatched_query():
    docs = [doc("肺炎患者出现呼吸困难需要紧急评估", "a"),
            doc("治疗取决于药物过敏和疾病严重程度", "b")]
    assert retriever.sparse_search("药物过敏", docs, 2)[0].metadata["passage_id"] == "b"
    assert retriever.sparse_search("xyzunmatched", docs, 2) == []


def test_rewrite_keeps_current_turn_and_passes_history():
    seen = []
    def model(messages):
        seen.extend(messages)
        return "肺炎患者出现哪些情况需要紧急评估？"
    result = retriever.rewrite_query("那什么时候要急诊？", [
        {"role": "user", "content": "我想了解肺炎"}], invoke=model)
    assert "肺炎" in result
    assert "那什么时候要急诊" in seen[-1].content
    assert "我想了解肺炎" in seen[-1].content
    assert retriever.rewrite_query("肺炎如何治疗", [], invoke=model) == "肺炎如何治疗"


def test_rerank_uses_pair_scores():
    class Encoder:
        def predict(self, pairs):
            assert pairs == [("query", "wrong"), ("query", "right")]
            return [-2.0, 4.0]
    result = retriever.rerank("query", [doc("wrong", "a"), doc("right", "b")], 1,
                              encoder=Encoder())
    assert result[0].metadata["passage_id"] == "b"
    assert result[0].metadata["rerank_score"] == 4.0


def test_rewrite_discards_qwen_orphan_thinking_tag():
    output = "reasoning " * 1000 + "</think>\n\n肺炎何时需要急诊？"
    assert retriever.rewrite_query("何时急诊？", [{"role": "user", "content": "肺炎"}],
                                   invoke=lambda _: output) == "肺炎何时需要急诊？"


def test_rewrite_rejects_empty_model_response():
    with pytest.raises(ValueError, match="empty"):
        retriever.rewrite_query("它呢", [{"role": "user", "content": "肺炎"}], invoke=lambda _: "")


def test_rewrite_rejects_truncated_reasoning():
    response = AIMessage(content="Okay, let us think about pneumonia", response_metadata={"done_reason": "length"})
    with pytest.raises(ValueError, match="truncated"):
        retriever.rewrite_query("它呢", [{"role": "user", "content": "肺炎"}], invoke=lambda _: response)


def test_hybrid_pipeline_uses_rewrite_and_rerank(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "PERSIST_DIRECTORY", tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "get_embeddings", FixedEmbeddings)
    store = vectorstore.get_vectorstore()
    store.add_documents([doc("肺炎诊断", "a"), doc("肺炎药物过敏治疗", "b")], ids=["a", "b"])
    monkeypatch.setattr(retriever, "rewrite_query", lambda query, history: "肺炎药物过敏")
    class Encoder:
        def predict(self, pairs):
            return [10.0 if "治疗" in text else 0.0 for _, text in pairs]
    monkeypatch.setattr(retriever, "get_reranker", lambda: Encoder())
    result = retriever.retrieve_guidelines("那怎么治", 1, history=[{"role": "user", "content": "肺炎"}])
    assert result[0].metadata["passage_id"] == "b"
    assert result[0].metadata["retrieval_query"] == "肺炎药物过敏"
    assert "rrf_score" in result[0].metadata


def test_build_is_idempotent_and_removes_stale_passages(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "PERSIST_DIRECTORY", tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "get_embeddings", FixedEmbeddings)
    documents = [Document(page_content="## Diagnosis\nCough", metadata={"source": "pneumonia.md"})]
    monkeypatch.setattr(vectorstore, "load_guidelines", lambda: documents)
    store = vectorstore.build_vectorstore()
    vectorstore.build_vectorstore()
    assert store._collection.count() == 1
    documents[0].page_content = "## Treatment\nAllergies"
    vectorstore.build_vectorstore()
    assert store.get()["ids"] == ["pneumonia.md#treatment-0"]


@pytest.mark.parametrize("query,k", [(" ", 1), ("test", 0), ("test", -1)])
def test_invalid_input(query, k):
    with pytest.raises(ValueError):
        retriever.retrieve_guidelines(query, k)


def test_empty_store_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "PERSIST_DIRECTORY", tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "get_embeddings", FixedEmbeddings)
    with pytest.raises(ValueError, match="build"):
        retriever.retrieve_guidelines("pneumonia")


@pytest.mark.parametrize("split", ["dev", "test"])
def test_candidate_labels_are_complete_and_honest(split):
    dataset = json.loads(Path(f"data/evaluation/{split}.json").read_text())
    cases = dataset["queries"]
    ids = {d.metadata["passage_id"] for d in split_documents(vectorstore.load_guidelines())}
    assert len(cases) == len({row["id"] for row in cases}) == 120
    assert len({row["query"] for row in cases}) == 120
    assert sum(bool(row["history"]) for row in cases) >= 20
    assert sum(len(row["relevant_passage_ids"]) > 1 for row in cases) >= 50
    assert all(set(row["relevant_passage_ids"]) <= ids and row["relevant_passage_ids"] for row in cases)
    for row in cases:
        assert row["annotation_status"] in {"pending_human_review", "human_reviewed"}
        if row["annotation_status"] == "human_reviewed":
            assert row["reviewer"] and row["reviewed_at"]
        else:
            assert row["reviewer"] is None
