"""FastAPI app: document assistant + voice client simulator."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

from . import llm  # noqa: E402
from .assistant import Assistant  # noqa: E402
from .knowledge import build_knowledge_base  # noqa: E402
from .simulator import PERSONAS, Simulator  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

kb = build_knowledge_base()
assistant = Assistant(kb)
simulator = Simulator(kb)

app = FastAPI(title="Advisor Voice Coach")


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class StartIn(BaseModel):
    persona_id: str


class TurnIn(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=1000)


class SessionIn(BaseModel):
    session_id: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"llm": llm.available(), "retrieval": kb.mode, "chunks": len(kb.chunks)}


@app.post("/api/ask")
def ask(body: AskIn):
    return assistant.ask(body.question).to_dict()


@app.get("/api/personas")
def personas():
    return [{"id": k, "title": p["title"], "client": p["client"], "mood": p["mood"]} for k, p in PERSONAS.items()]


@app.post("/api/sim/start")
def sim_start(body: StartIn):
    if body.persona_id not in PERSONAS:
        raise HTTPException(404, "unknown persona")
    return simulator.start(body.persona_id)


@app.post("/api/sim/turn")
def sim_turn(body: TurnIn):
    if body.session_id not in simulator.sessions:
        raise HTTPException(404, "unknown session")
    return simulator.turn(body.session_id, body.text)


@app.post("/api/sim/debrief")
def sim_debrief(body: SessionIn):
    if body.session_id not in simulator.sessions:
        raise HTTPException(404, "unknown session")
    return simulator.debrief(body.session_id)
