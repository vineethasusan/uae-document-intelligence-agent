"""LLM providers behind one small interface, using only the standard library (no SDK lock-in).

Set DOCAGENT_LLM to one of: openai, azure, anthropic. Without it the agent runs fully
offline with the rule engine and extractive answers, which is also what the tests use.

    openai    -> OPENAI_API_KEY, optional OPENAI_MODEL (default gpt-4o-mini)
    azure     -> AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT
    anthropic -> ANTHROPIC_API_KEY, optional ANTHROPIC_MODEL
"""
from __future__ import annotations

import json
import os
import urllib.request

from .generate import LABELS

SYSTEM = ("You extract fields from UAE business documents written in Arabic, English or both. "
          "Return only JSON. Copy values exactly as written, converting Arabic-Indic digits to Western digits. "
          "Use null when a field is not present. Never invent values.")


def extraction_prompt(text: str, doc_type: str) -> str:
    keys = [k for k in LABELS[doc_type] if k != "title"]
    hints = {k: f"{LABELS[doc_type][k][0]} / {LABELS[doc_type][k][1]}" for k in keys}
    return (f"Document type: {doc_type}\nReturn a JSON object with exactly these keys: {json.dumps(hints, ensure_ascii=False)}\n"
            f"---\n{text}\n---")


def answer_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return (f"Answer the question using only the passages. Cite passage numbers like [1]. "
            f"If the answer is not in the passages, say you cannot find it.\n\n{ctx}\n\nQuestion: {question}")


def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def complete(prompt: str, system: str = SYSTEM, json_mode: bool = False) -> str:
    provider = os.getenv("DOCAGENT_LLM", "").lower()
    if provider == "openai":
        body = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = _post("https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, body)
        return r["choices"][0]["message"]["content"]
    if provider == "azure":
        url = (f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/deployments/{os.environ['AZURE_OPENAI_DEPLOYMENT']}"
               f"/chat/completions?api-version=2024-06-01")
        body = {"temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return _post(url, {"api-key": os.environ["AZURE_OPENAI_KEY"]}, body)["choices"][0]["message"]["content"]
    if provider == "anthropic":
        body = {"model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"), "max_tokens": 1024, "temperature": 0,
                "system": system, "messages": [{"role": "user", "content": prompt}]}
        r = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}, body)
        return r["content"][0]["text"]
    raise RuntimeError("No LLM configured (set DOCAGENT_LLM); the agent will use its offline engines.")


def llm_available() -> bool:
    return os.getenv("DOCAGENT_LLM", "").lower() in {"openai", "azure", "anthropic"}


def llm_extract(text: str, doc_type: str) -> dict:
    raw = complete(extraction_prompt(text, doc_type), json_mode=True)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(raw)
