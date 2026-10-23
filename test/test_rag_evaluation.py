import pytest

from src.rag.evaluation import retrieval_metrics, select_queries
from src.rag import evaluation
from langchain_core.documents import Document


def test_metrics_use_rank_and_ignore_duplicate_hits():
    result = retrieval_metrics(["wrong", "a", "a", "b"], {"a", "b"}, 3)
    assert result["recall"] == 0.5
    assert result["hit_rate"] == 1.0
    assert result["mrr"] == 0.5
    assert result["ndcg"] == pytest.approx((1 / __import__('math').log2(3)) / (1 + 1 / __import__('math').log2(3)))


def test_annotation_filter_cannot_call_synthetic_cases_human_reviewed():
    cases = [{"annotation_status": "pending_human_review", "reviewer": None},
             {"annotation_status": "human_reviewed", "reviewer": "Reviewer A"}]
    assert select_queries(cases, include_unreviewed=False) == cases[1:]
    assert select_queries(cases, include_unreviewed=True) == cases
    with pytest.raises(ValueError, match="reviewer"):
        select_queries([{"annotation_status": "human_reviewed", "reviewer": None}], False)


def test_evaluation_rejects_changed_corpus_with_same_passage_id():
    docs = [Document(page_content="original", metadata={"passage_id": "a"})]
    cases = [{"id": "query-1", "corpus_sha256": evaluation.corpus_fingerprint(docs)}]
    evaluation.validate_corpus(cases, docs)
    docs[0].page_content = "changed clinical recommendation"
    with pytest.raises(ValueError, match="corpus"):
        evaluation.validate_corpus(cases, docs)


def test_recall_differs_from_hit_rate_for_multiple_gold_passages():
    result = retrieval_metrics(["a", "x", "y", "z", "w"], {"a", "b", "c"}, 5)
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["hit_rate"] == 1.0
