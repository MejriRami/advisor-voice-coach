"""Offline tests (no API key): retrieval, guardrails, evaluation gate, API."""
import os

# Force offline mode even when a local .env holds a key (load_dotenv never overrides).
os.environ["OPENAI_API_KEY"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.assistant import HANDOFF_MESSAGE, Assistant  # noqa: E402
from app.knowledge import KnowledgeBase, load_chunks  # noqa: E402
from app.simulator import PERSONAS, keyword_debrief  # noqa: E402
from eval.run_eval import evaluate, load_questions  # noqa: E402


@pytest.fixture(scope="module")
def assistant():
    return Assistant(KnowledgeBase())


def test_every_document_is_split_into_sections():
    chunks = load_chunks()
    assert len({c.doc for c in chunks}) == 6
    assert all(c.section and c.text for c in chunks)
    assert all("fictif" not in c.doc_title for c in chunks)


@pytest.mark.parametrize("question, doc", [
    ("délai pour déclarer un dégât des eaux", "02_habitation"),
    ("plafond lunettes formule Confort", "04_sante_complementaire"),
    ("frais de fractionnement paiement mensuel", "06_faq_paiement"),
])
def test_retrieval_finds_the_right_document(assistant, question, doc):
    assert assistant.kb.search(question, 1)[0][0].doc == doc


def test_answers_are_cited(assistant):
    a = assistant.ask("Quel est le délai de carence pour la maternité ?")
    assert not a.handoff
    assert "9 mois" in a.text
    assert a.citations and a.citations[0]["doc"] == "04_sante_complementaire"


@pytest.mark.parametrize("question", [
    "Quel est le tarif exact d'une assurance Tous Risques pour une Peugeot 208 ?",
    "Pouvez-vous me conseiller un placement en bourse ?",
    "Quelle est la météo prévue demain à Tunis ?",
])
def test_out_of_scope_questions_go_to_a_human(assistant, question):
    a = assistant.ask(question)
    assert a.handoff and a.text == HANDOFF_MESSAGE and not a.citations


def test_offline_evaluation_meets_its_gate(assistant):
    report = evaluate(assistant, load_questions())
    assert report["retrieval_hit@3"] == 100.0
    assert report["precision_when_answering"] >= 90.0
    assert report["handoff_on_out_of_scope"] == 100.0
    assert report["citation_rate"] == 100.0


def test_keyword_debrief_scores_the_checklist():
    checklist = PERSONAS["accident_auto"]["checklist"]
    good = ("Vous avez 5 jours ouvrés pour envoyer le constat. Au-delà de 1 500 TND un expert passe. "
            "Avec moins de 3 ans de permis la franchise est doublée. Je ne peux pas promettre l'indemnisation avant l'expertise.")
    assert keyword_debrief(checklist, good)["score"] > keyword_debrief(checklist, "Ne vous inquiétez pas.")["score"]


def test_api_end_to_end():
    from app.main import app
    client = TestClient(app)
    assert client.get("/api/health").json()["llm"] is False
    assert client.post("/api/ask", json={"question": "Combien coûte la franchise bris de glace ?"}).json()["handoff"] is False
    started = client.post("/api/sim/start", json={"persona_id": "client_mecontent"}).json()
    assert started["reply"]
    client.post("/api/sim/turn", json={"session_id": started["session_id"], "text": "Je comprends, je suis désolé."})
    debrief = client.post("/api/sim/debrief", json={"session_id": started["session_id"]}).json()
    assert 0 <= debrief["score"] <= 100 and debrief["criteria"]
    assert client.post("/api/sim/start", json={"persona_id": "nope"}).status_code == 404
