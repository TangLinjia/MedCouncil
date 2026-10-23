import json

from langchain.tools import tool
from pydantic import BaseModel, Field

from src.rag.retriever import retrieve_guidelines


class GuidelineSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Clinical guideline question to search for, "
            "including the condition and relevant patient context."
        )
    )

    top_k: int = Field(
        default=4,
        ge=1,
        le=6,
        description="Number of guideline passages to retrieve.",
    )

    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior user/assistant turns with role and content, for resolving references and omitted context.",
    )


@tool("guideline_search", args_schema=GuidelineSearchInput)
def guideline_search(query: str, top_k: int = 4, history: list[dict[str, str]] | None = None) -> str:
    """
    Search the local clinical guideline knowledge base.

    Use this tool when evidence from clinical guidelines is needed
    for diagnosis, recommended testing, treatment, monitoring,
    follow-up, or precautions.
    """

    documents = retrieve_guidelines(
        query=query,
        k=top_k,
        history=history,
    )

    results = []

    for index, document in enumerate(documents, start=1):
        results.append(
            {
                "rank": index,
                "passage_id": document.metadata.get("passage_id"),
                "source_url": document.metadata.get("source_url"),
                "retrieval_query": document.metadata.get("retrieval_query"),
                "rrf_score": document.metadata.get("rrf_score"),
                "rerank_score": document.metadata.get("rerank_score"),
                "content": document.page_content,
                "source": document.metadata.get(
                    "source",
                    "unknown",
                ),
                "condition": document.metadata.get(
                    "condition",
                ),
            }
        )

    print(f"[TOOL] guideline_search called: {query}")

    return json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
    )
