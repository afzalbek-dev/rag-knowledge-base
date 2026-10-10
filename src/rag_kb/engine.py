import math
import re
from typing import List, Dict, Tuple
from collections import Counter
from .chunker import Chunk


class TFIDFSearchEngine:
    """Tashqi og'ir kutubxonalarsiz sof Python TF-IDF qidiruv mexanizmi."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Harflar va raqamlarni ajratib kichik harfga o'tkazish
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_'\u0400-\u04FF\u0100-\u017F]+\b", text)]

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        num_docs = len(chunks)
        if num_docs == 0:
            return

        doc_frequencies: Dict[str, int] = Counter()
        tokenized_docs = []

        for chunk in chunks:
            tokens = self.tokenize(chunk.content)
            tokenized_docs.append(tokens)
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_frequencies[term] += 1

        # Lug'at va IDF hisoblash (log(N / DF))
        self.idf = {term: math.log((num_docs + 1) / (df + 1)) + 1.0 for term, df in doc_frequencies.items()}

        # Hujjatlar uchun TF-IDF vektorlarini qurish
        self.doc_vectors = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            doc_len = len(tokens) or 1
            vec = {}
            for term, count in tf.items():
                vec[term] = (count / doc_len) * self.idf.get(term, 0.0)
            self.doc_vectors.append(vec)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if not self.chunks or not self.doc_vectors:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        q_tf = Counter(query_tokens)
        q_len = len(query_tokens)
        q_vec = {term: (count / q_len) * self.idf.get(term, 0.0) for term, count in q_tf.items()}

        q_norm = math.sqrt(sum(val * val for val in q_vec.values()))
        if q_norm == 0:
            return []

        scores: List[Tuple[Chunk, float]] = []
        for i, d_vec in enumerate(self.doc_vectors):
            dot_product = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
            d_norm = math.sqrt(sum(val * val for val in d_vec.values()))
            similarity = (dot_product / (q_norm * d_norm)) if d_norm > 0 else 0.0

            if similarity > 0:
                scores.append((self.chunks[i], similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
