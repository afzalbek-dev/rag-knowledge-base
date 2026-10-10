import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: str
    source_file: str
    content: str
    chunk_index: int


class DocumentChunker:
    """Hujjatlarni semantik bo'laklarga (chunks) ajratuvchi modul."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source_file: str = "doc") -> List[Chunk]:
        # Paragraflarga ajratish
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[Chunk] = []

        current_chunk = []
        current_len = 0
        chunk_idx = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self.chunk_size and current_chunk:
                combined_text = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        id=f"{source_file}:{chunk_idx}",
                        source_file=source_file,
                        content=combined_text,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
                current_chunk = [para]
                current_len = para_len
            else:
                current_chunk.append(para)
                current_len += para_len + 2

        if current_chunk:
            combined_text = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    id=f"{source_file}:{chunk_idx}",
                    source_file=source_file,
                    content=combined_text,
                    chunk_index=chunk_idx,
                )
            )

        return chunks
