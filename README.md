# RAG Knowledge Base

Mahalliy Markdown va matnli hujjatlar uchun tashqi og'ir paketlarsiz (PyTorch/Transformers talab qilmaydigan) sof Python TF-IDF va kosinus o'xshashlikka asoslangan RAG qidiruv tizimi.

## Xususiyatlari
- **0 Og'ir Bog'liqliklar**: RAM va xotirani tejaydi, Termux va ARM64 serverlarda soniyalarda ishga tushadi.
- **Semantik Paragraf Chunking**: Matnlarni mantiqiy bloklarga ajratish va manba faylini kuzatish.
- **TF-IDF & Cosine Similarity**: Qidiruv so'roviga eng mos hujjat parchalarini tezkor saralash.
- **LLM Kontekst Tayyorlash**: Topilgan natijalarni manbalar va ballar bilan birga tayyor kontekst sifatida chiqarish.

## Ishlatish (Python)
```python
from rag_kb import KnowledgeBase

kb = KnowledgeBase()
kb.ingest_directory("./my_docs")

result = kb.query("Qanday qilib deploy qilish mumkin?", top_k=2)
print("Topilgan kontekst:\n", result["context"])
```

## CLI
```bash
rag-kb --dir ./docs --query "Ma'lumotlar bazasi arxitekturasi"
```

## Testlar
```bash
pytest tests/ -v
```

## Litsenziya
MIT
