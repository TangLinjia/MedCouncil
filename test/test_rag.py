from src.rag.loader import load_guidelines
from src.rag.chunker import split_documents
from src.rag.vectorstore import build_vectorstore
from src.rag.retriever import retrieve_guidelines
from src.tools.guideline_search import guideline_search


print("\n=== 1. TEST LOADER ===")

documents = load_guidelines()

print(f"Loaded documents: {len(documents)}")

for doc in documents:
    print(doc.metadata)
    print(doc.page_content[:100])


print("\n=== 2. TEST CHUNKER ===")

chunks = split_documents(documents)

print(f"Generated chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}")
    print(chunk.metadata)
    print(chunk.page_content[:200])


print("\n=== 3. BUILD VECTOR STORE ===")

build_vectorstore()

print("Vector store built successfully.")


print("\n=== 4. TEST RETRIEVAL ===")

results = retrieve_guidelines(
    "What warning signs require urgent evaluation in pneumonia?",
    k=3,
)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Metadata:", doc.metadata)
    print("Content:", doc.page_content)


print("\n=== 5. TEST TOOL ===")

tool_result = guideline_search.invoke(
    {
        "query": (
            "What warning signs require urgent medical "
            "evaluation in pneumonia?"
        ),
        "top_k": 3,
    }
)

print(tool_result)
