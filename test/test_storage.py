from langchain_core.embeddings import Embeddings

from src.rag import vectorstore
from src.tools.drug_safety_check import load_rules


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]




def test_vectorstore_reopens_persisted_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, 'PERSIST_DIRECTORY', tmp_path / 'chroma')
    monkeypatch.setattr(vectorstore, 'get_embeddings', FixedEmbeddings)
    store = vectorstore.get_vectorstore()
    store.add_texts(['持久化指南内容'], ids=['persistence-test'])
    reopened = vectorstore.get_vectorstore()
    assert reopened.get(ids=['persistence-test'])['documents'] == ['持久化指南内容']
    assert (tmp_path / 'chroma' / 'chroma.sqlite3').exists()


def test_drug_rules_are_loaded_once():
    load_rules.cache_clear()
    try:
        first = load_rules()
        assert load_rules() is first
        assert load_rules.cache_info().misses == 1
        assert load_rules.cache_info().hits == 1
        assert first['interaction_index']
    finally:
        load_rules.cache_clear()
