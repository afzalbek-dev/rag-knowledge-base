import argparse
import sys
import json
import os
from .knowledge_base import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(description="RAG Knowledge Base - Mahalliy Markdown hujjatlar qidiruvi")
    parser.add_argument("--dir", required=True, help="Hujjatlar papkasi (.md yoki .txt)")
    parser.add_argument("--query", required=True, help="Qidiruv savoli")
    parser.add_argument("--top", type=int, default=3, help="Qaytariladigan eng yaqin bo'laklar soni")
    parser.add_argument("--json", action="store_true", help="JSON formatida chiqarish")

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Papka topilmadi: {args.dir}", file=sys.stderr)
        sys.exit(1)

    kb = KnowledgeBase()
    count = kb.ingest_directory(args.dir)
    res = kb.query(args.query, top_k=args.top)

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(f"🔎 Indekslangan bo'laklar: {count} ta")
        print(f"Savol: {args.query}\n")
        print("Topilgan kontekst:")
        print(res["context"])


if __name__ == "__main__":
    main()
