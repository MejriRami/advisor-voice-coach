"""Answers advisors' questions from the documents only, with citations, or hands off to a human.

Guardrails (in code, not only in the prompt):
1. Retrieval gate: if the best passage is below a confidence threshold, no answer is generated.
2. Grounding: the model only sees the retrieved passages and must cite them as [1], [2]...
3. Citation check: an answer without a valid citation is discarded and replaced by a handoff.
4. Explicit refusal: the model returns HANDOFF when the passages do not contain the answer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from . import llm
from .knowledge import Chunk, KnowledgeBase, TfidfEmbedder

HANDOFF_MESSAGE = ("Je ne trouve pas cette information dans nos documents. "
                   "Je préfère ne pas vous répondre approximativement : je transmets votre question à un conseiller.")

# Cosine thresholds differ by retriever; both were tuned on eval/questions.jsonl (see README).
THRESHOLDS = {"tfidf": 0.12, "openai": 0.30}

SYSTEM_PROMPT = """Tu es l'assistant interne des conseillers de Nova Assurances.
Réponds en français, en 1 à 3 phrases claires, UNIQUEMENT à partir des extraits numérotés fournis.
Cite chaque information avec le numéro de l'extrait entre crochets, par exemple [1].
N'invente jamais un montant, un délai ou une garantie. Ne donne pas de tarif.
Si les extraits ne permettent pas de répondre avec certitude, réponds exactement : HANDOFF"""


@dataclass
class Answer:
    text: str
    handoff: bool
    citations: list[dict] = field(default_factory=list)
    top_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {"answer": self.text, "handoff": self.handoff, "citations": self.citations,
                "top_score": round(self.top_score, 3), "reason": self.reason}


MIN_COVERAGE = 0.5
_QUESTION_WORDS = {"combie", "quand", "commen", "quel", "quelle", "faut", "peut", "doit", "partir", "jusqu",
                   "client", "est", "sont", "temps", "delai", "montan", "existe", "propos", "pouvez"}


def coverage(question: str, chunk: Chunk) -> float:
    """Share of the question's content words that appear in the passage (offline scope check)."""
    words = {w for w in TfidfEmbedder.tokens(question) if w not in _QUESTION_WORDS}
    if not words:
        return 1.0
    passage = set(TfidfEmbedder.tokens(f"{chunk.doc_title} {chunk.section} {chunk.text}"))
    return len(words & passage) / len(words)


def _context(passages: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{i}] {c.citation}\n{c.text}" for i, (c, _) in enumerate(passages, 1))


def _cited_numbers(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})


def _extractive(question: str, passages: list[tuple[Chunk, float]]) -> str:
    """Offline fallback: the sentences of the best passage that share most words with the question."""
    q = set(TfidfEmbedder.tokens(question))
    chunk = passages[0][0]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", chunk.text) if s.strip()]
    ranked = sorted(sentences, key=lambda s: -len(q & set(TfidfEmbedder.tokens(s))))
    best = [s.lstrip("- ").replace("**", "") for s in ranked[:2]]
    return " ".join(best) + " [1]"


class Assistant:
    def __init__(self, kb: KnowledgeBase, k: int = 4):
        self.kb = kb
        self.k = k
        self.threshold = float(os.getenv("HANDOFF_THRESHOLD", THRESHOLDS[kb.mode]))

    def ask(self, question: str) -> Answer:
        passages = self.kb.search(question, self.k)
        top = passages[0][1] if passages else 0.0
        if not passages or top < self.threshold:
            return Answer(HANDOFF_MESSAGE, True, top_score=top, reason="low_retrieval_score")
        if self.kb.mode == "tfidf" and coverage(question, passages[0][0]) < MIN_COVERAGE:
            # Keyword retrieval can't judge scope on its own (one shared word scores high),
            # so offline mode also requires most of the question's key words in the passage.
            return Answer(HANDOFF_MESSAGE, True, top_score=top, reason="low_question_coverage")

        if llm.available():
            raw = llm.chat(SYSTEM_PROMPT, [{"role": "user", "content":
                           f"Extraits :\n{_context(passages)}\n\nQuestion du conseiller : {question}"}]).strip()
            if raw.upper().startswith("HANDOFF"):
                return Answer(HANDOFF_MESSAGE, True, top_score=top, reason="model_declined")
        else:
            raw = _extractive(question, passages)

        cited = _cited_numbers(raw, len(passages))
        if not cited:
            return Answer(HANDOFF_MESSAGE, True, top_score=top, reason="no_citation")

        citations = [{"n": n, "source": passages[n - 1][0].citation, "doc": passages[n - 1][0].doc} for n in cited]
        return Answer(raw, False, citations, top)
