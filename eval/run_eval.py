"""Measure the assistant on eval/questions.jsonl.

Metrics
- retrieval hit@1 / hit@3: the expected document is the first / among the first 3 passages
- answer accuracy: in-scope answers that contain every expected fact (and are not a handoff)
- precision when answering: of the in-scope questions it did answer, how many were right
  (for an insurer, a wrong answer is worse than a handoff to a human)
- citation rate: answers (not handoffs) that cite at least one passage
- handoff on out-of-scope: out-of-scope questions correctly handed to a human
- false handoffs: in-scope questions wrongly handed off

    python -m eval.run_eval            # uses OpenAI if OPENAI_API_KEY is set, else offline TF-IDF
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import llm  # noqa: E402
from app.assistant import Assistant  # noqa: E402
from app.knowledge import build_knowledge_base, normalize  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contains_all(answer: str, facts: list[str]) -> bool:
    text = normalize(answer).replace(" ", " ").replace("\xa0", " ")
    return all(normalize(f) in text for f in facts)


def evaluate(assistant: Assistant, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        passages = assistant.kb.search(q["question"], 3)
        answer = assistant.ask(q["question"])
        docs = [c.doc for c, _ in passages]
        rows.append({
            "id": q["id"], "out_of_scope": q.get("out_of_scope", False),
            "hit1": bool(docs) and docs[0] == q["expected_doc"],
            "hit3": q["expected_doc"] in docs,
            "handoff": answer.handoff,
            "correct": (not answer.handoff) and contains_all(answer.text, q["must_contain"]),
            "cited": bool(answer.citations),
            "answer": answer.text, "reason": answer.reason, "top_score": round(answer.top_score, 3),
        })

    ins = [r for r in rows if not r["out_of_scope"]]
    outs = [r for r in rows if r["out_of_scope"]]
    pct = lambda n, d: round(100 * n / d, 1) if d else 0.0  # noqa: E731
    return {
        "mode": f"{assistant.kb.mode} retrieval, {'LLM' if llm.available() else 'extractive (no LLM)'} answers",
        "in_scope": len(ins), "out_of_scope": len(outs),
        "retrieval_hit@1": pct(sum(r["hit1"] for r in ins), len(ins)),
        "retrieval_hit@3": pct(sum(r["hit3"] for r in ins), len(ins)),
        "answer_accuracy": pct(sum(r["correct"] for r in ins), len(ins)),
        "precision_when_answering": pct(sum(r["correct"] for r in ins), sum(not r["handoff"] for r in ins)),
        "citation_rate": pct(sum(r["cited"] for r in ins if not r["handoff"]), sum(not r["handoff"] for r in ins)),
        "false_handoffs": sum(r["handoff"] for r in ins),
        "handoff_on_out_of_scope": pct(sum(r["handoff"] for r in outs), len(outs)),
        "rows": rows,
    }


def main() -> None:
    started = time.time()
    report = evaluate(Assistant(build_knowledge_base()), load_questions())
    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = [r for r in report["rows"] if (r["out_of_scope"] and not r["handoff"]) or (not r["out_of_scope"] and not r["correct"])]
    for r in failures:
        print(f"  ✗ {r['id']} handoff={r['handoff']} ({r['reason'] or '-'}) score={r['top_score']}: {r['answer'][:110]}")
    out = ROOT / "eval" / "last_results.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(report['rows'])} questions in {time.time() - started:.1f}s → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
