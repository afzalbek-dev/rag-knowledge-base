import pytest
from rag_kb.chunker import DocumentChunker
from rag_kb.engine import TFIDFSearchEngine
from rag_kb.knowledge_base import KnowledgeBase


def test_chunker_paragraphs():
    text = (
        "Birinchi paragraf: Python dasturlash tili haqida.\n\n"
        "Ikkinchi paragraf: Sun'iy intellekt va mashinali o'rganish.\n\n"
        "Uchinchi paragraf: FastAPI va veb texnologiyalar."
    )
    chunker = DocumentChunker(chunk_size=100)
    chunks = chunker.chunk_text(text, source_file="intro.md")
    assert len(chunks) >= 2
    assert "Python" in chunks[0].content


def test_tfidf_search_accuracy():
    chunker = DocumentChunker()
    docs = [
        "PostgreSQL - kuchli reliesion ma'lumotlar bazasi tizimi.",
        "Redis - tezkor xotirada ishlovchi kalit-qiymat ombori va kesh tizimi.",
        "Docker - ilovalarni konteynerlashtirish va izolyatsiya qilish platformasi.",
    ]
    chunks = []
    for i, d in enumerate(docs):
        chunks.extend(chunker.chunk_text(d, source_file=f"doc_{i}.md"))

    engine = TFIDFSearchEngine()
    engine.index(chunks)

    # Redis haqida qidiruv
    results = engine.search("kesh tizimi va Redis", top_k=1)
    assert len(results) == 1
    top_chunk, score = results[0]
    assert "Redis" in top_chunk.content
    assert score > 0.0


def test_knowledge_base_dir_ingestion(tmp_path):
    doc1 = tmp_path / "guide.md"
    doc1.write_text("# Python Qo'llanma\n\nPython juda sodda va o'qilishi oson dasturlash tili hisoblanadi.")

    doc2 = tmp_path / "arch.md"
    doc2.write_text("# Arxitektura\n\nClean architecture tamoyillari loyiha barqarorligini ta'minlaydi.")

    kb = KnowledgeBase()
    count = kb.ingest_directory(str(tmp_path))
    assert count >= 2

    res = kb.query("Clean architecture")
    assert "barqarorligini" in res["context"]
    assert len(res["sources"]) > 0
