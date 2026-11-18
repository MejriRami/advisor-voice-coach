"""Thin wrapper around the chat model, so the rest of the code stays provider-agnostic."""
from __future__ import annotations

import json
import os


def available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def chat(system: str, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.2) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        messages=[{"role": "system", "content": system}, *messages],
        response_format={"type": "json_object"} if json_mode else None,
    )
    return response.choices[0].message.content or ""


def chat_json(system: str, messages: list[dict]) -> dict:
    return json.loads(chat(system, messages, json_mode=True, temperature=0))
