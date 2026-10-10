import os
from typing import List, Dict, Any
from .chunker import DocumentChunker, Chunk
from .engine import TFIDFSearchEngine


class KnowledgeBase:
    """Markdown fayllarini o'qish, indekslash va savol-javob kontekstini yig'ish."""

    def __init__(self, chunk_size: int = 500):
        self.chunker = DocumentChunker(chunk_size=chunk_size)
        self.engine = TFIDFSearchEngine()

    def ingest_directory(self, dir_path: str) -> int:
        all_chunks: List[Chunk] = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".md", ".txt")):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, dir_path)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        chunks = self.chunker.chunk_text(text, source_file=rel_path)
                        all_chunks.extend(chunks)
                    except Exception:
                        continue

        self.engine.index(all_chunks)
        return len(all_chunks)

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        results = self.engine.search(question, top_k=top_k)
        context_parts = []
        sources = []

        for chunk, score in results:
            context_parts.append(f"[{chunk.source_file} (Score: {score:.3f})]:\n{chunk.content}")
            sources.append({"source": chunk.source_file, "score": round(score, 3)})

        return {
            "question": question,
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
        }
