"""Knowledge base: load documents, split them into sections, retrieve the relevant ones.

Two interchangeable embedders:
- OpenAIEmbedder: semantic embeddings (needs OPENAI_API_KEY)
- TfidfEmbedder: local, offline, deterministic (tests, CI, demo without a key)
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str          # file stem, e.g. "02_habitation"
    doc_title: str    # "Assurance Habitation — Conditions générales"
    section: str      # "Dégât des eaux"
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_title} › {self.section}"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """One chunk per '## ' section; the section keeps its document title for context."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", raw, re.M)
        doc_title = re.sub(r"\s*\(.*?fictif\)\s*$", "", title_match.group(1)).strip() if title_match else path.stem
        for i, block in enumerate(re.split(r"^## ", raw, flags=re.M)[1:]):
            heading, _, body = block.partition("\n")
            body = body.strip()
            if body:
                chunks.append(Chunk(f"{path.stem}#{i}", path.stem, doc_title, heading.strip(), body))
    return chunks


class TfidfEmbedder:
    """Small dependency-free TF-IDF with accent folding and light French stemming."""

    _STOP = set("le la les un une des du de d l et ou a au aux en dans pour par sur avec sans est sont "
                "ce cet cette ces il elle ils on qui que quoi quel quelle quels quelles se sa son ses "
                "mon ma mes ne pas plus je j me m y vous nous leur leurs si comme etre avoir fait".split())

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    @classmethod
    def tokens(cls, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", normalize(text))
        return [w[:6] for w in words if w not in cls._STOP and len(w) > 1]  # crude stem: 6-char prefix

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        df: dict[str, int] = {}
        for t in texts:
            for tok in set(self.tokens(t)):
                df[tok] = df.get(tok, 0) + 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[tok])) + 1 for tok in sorted(df)])
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.vocab)))
        for row, t in enumerate(texts):
            for tok in self.tokens(t):
                if tok in self.vocab:
                    out[row, self.vocab[tok]] += 1
        out *= self.idf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class OpenAIEmbedder:
    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def fit(self, texts: list[str]) -> "OpenAIEmbedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        data = self.client.embeddings.create(model=self.model, input=texts).data
        vectors = np.array([d.embedding for d in data])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk] | None = None, embedder=None):
        self.chunks = chunks if chunks is not None else load_chunks()
        texts = [self._indexed_text(c) for c in self.chunks]
        self.embedder = (embedder or TfidfEmbedder()).fit(texts)
        self.matrix = self.embedder.embed(texts)

    @staticmethod
    def _indexed_text(chunk: Chunk) -> str:
        return f"{chunk.doc_title}. {chunk.section}. {chunk.text}"

    @property
    def mode(self) -> str:
        return "openai" if isinstance(self.embedder, OpenAIEmbedder) else "tfidf"

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def build_knowledge_base() -> KnowledgeBase:
    use_openai = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("RETRIEVAL", "openai") == "openai"
    return KnowledgeBase(embedder=OpenAIEmbedder() if use_openai else TfidfEmbedder())
