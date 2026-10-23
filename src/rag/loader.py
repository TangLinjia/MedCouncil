from pathlib import Path
import json
from langchain_core.documents import Document


GUIDELINE_DIR = Path(__file__).resolve().parents[2] / "data" / "guidelines"


def load_guidelines(directory: Path = GUIDELINE_DIR,) -> list[Document]:
    """Load local clinical guideline documents."""

    documents: list[Document] = []

    if not directory.exists():
        raise FileNotFoundError(
            f"Guideline directory does not exist: {directory}"
        )

    manifest = directory / "sources.json"
    sources = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8").strip()

        if not text: continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    **sources.get(path.name, {}),
                    "source": path.name,
                    "condition": path.stem,
                },
            )
        )

    return documents
