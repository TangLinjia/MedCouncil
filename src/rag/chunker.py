import hashlib
import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def corpus_fingerprint(documents: list[Document]) -> str:
    snapshot = sorted((d.metadata["passage_id"], d.page_content) for d in documents)
    return hashlib.sha256(json.dumps(snapshot, ensure_ascii=False).encode("utf-8")).hexdigest()


def split_documents(documents: list[Document]) -> list[Document]:
    """Split guideline documents into retrieval-sized chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = []
    headers = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title"), ("##", "section")])
    for document in documents:
        sections = headers.split_text(document.page_content)
        counters = {}
        for section in sections:
            name = section.metadata.get("section", "overview")
            slug = re.sub(r"[^\w-]+", "-", name.lower()).strip("-") or "overview"
            for chunk in splitter.split_documents([section]):
                index = counters.get(slug, 0)
                counters[slug] = index + 1
                chunk.metadata = {**document.metadata, **chunk.metadata,
                                  "passage_id": f"{document.metadata['source']}#{slug}-{index}"}
                # Keep the section title in embeddings and lexical search.
                chunk.page_content = f"{section.metadata.get('title', '')} — {name}\n{chunk.page_content}"
                chunks.append(chunk)
    return chunks
