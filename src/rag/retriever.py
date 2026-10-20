from langchain_core.documents import Document

from .vectorstore import get_vectorstore


def retrieve_guidelines(query: str, k: int = 4) -> list[Document]:
    """Retrieve the most relevant guideline chunks."""

    if not query.strip():
        raise ValueError("Query must not be empty.")

    vectorstore = get_vectorstore()

    return vectorstore.similarity_search(query=query, k=k)
