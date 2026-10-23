import pytest

from src.rag import benchmark


def case(identifier, query, group):
    return {"id": identifier, "query": query, "history": [], "group_id": group,
            "relevant_passage_ids": [group + "#e1-0"], "annotation_status": "pending_human_review",
            "reviewer": None}


def test_split_audit_detects_contextual_duplicates_and_shared_groups():
    dev = [case("D1", "肺炎，如何检查？", "pneumonia")]
    test = [case("T1", "肺炎如何检查", "other")]
    report = benchmark.audit_splits(dev, test)
    assert report["exact_overlap"]
    report = benchmark.audit_splits(dev, [case("T2", "用药多久", "pneumonia")])
    assert report["shared_groups"] == ["pneumonia"]


def test_same_followup_with_different_antecedents_is_not_exact_duplicate():
    a, b = case("D1", "它呢", "a"), case("T1", "它呢", "b")
    a["history"] = [{"role": "user", "content": "肺炎"}]
    b["history"] = [{"role": "user", "content": "疟疾"}]
    assert not benchmark.audit_splits([a], [b])["exact_overlap"]


def test_freeze_rejects_different_code_or_data():
    freeze = {"code_sha256": "a", "dev_sha256": "b", "test_sha256": "c", "corpus_sha256": "d"}
    benchmark.check_frozen(freeze, dict(freeze))
    with pytest.raises(ValueError, match="code_sha256"):
        benchmark.check_frozen(freeze, {**freeze, "code_sha256": "changed"})


def test_semantic_audit_checks_both_splits_and_blocks_cross_split_neighbors():
    dev = [case("D1", "question A", "a"), case("D2", "question B", "b")]
    test = [case("T1", "question C", "c")]
    report = benchmark.audit_splits(dev, test, vectors=[[1., 0.], [0., 1.], [0.999, 0.001]])
    assert report["semantic_overlap"][0]["left"] == "D1"
    assert report["semantic_overlap"][0]["right"] == "T1"


def test_audit_identity_does_not_change_when_timestamp_changes():
    report = {"created_at": "first run", "passed": True, "semantic_overlap": []}
    assert benchmark.audit_signature(report) == benchmark.audit_signature({**report, "created_at": "second run"})


def test_local_model_fingerprint_changes_when_weight_changes(tmp_path):
    directory = tmp_path / ".cache" / "snapshot"
    directory.mkdir(parents=True)
    weights = directory / "model.safetensors"
    weights.write_bytes(b"first")
    first = benchmark.snapshot_digest(directory)
    weights.write_bytes(b"second")
    assert benchmark.snapshot_digest(directory) != first


def test_cross_source_question_cannot_hide_a_shared_topic():
    a, b = case("D1", "one", "a"), case("T1", "two", "b")
    b["source_group_ids"] = ["a", "b"]
    assert benchmark.audit_splits([a], [b])["shared_groups"] == ["a"]
