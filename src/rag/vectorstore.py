from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .loader import load_guidelines
from .chunker import split_documents


PERSIST_DIRECTORY = Path("data/chroma_db")
COLLECTION_NAME = "medical_guidelines"

EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )


def build_vectorstore() -> Chroma:
    """Build the guideline vector store from local documents."""

    documents = load_guidelines()
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No guideline documents were found.")

    PERSIST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    return vectorstore


def get_vectorstore() -> Chroma:
    """Load the existing persistent guideline vector store."""

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIRECTORY),
    )
