"""REST API: POST a document's text, get back its type, fields, checks and the engine used for each field."""
from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel

from docagent.agent import DocumentAgent
from docagent.generate import generate

TODAY = date.today()
_docs = generate(n=400, seed=7, today=TODAY)
agent = DocumentAgent(TODAY).fit([d.text for d in _docs], [d.doc_type for d in _docs])
app = FastAPI(title="UAE Document Intelligence Agent", version="1.0.0")


class Doc(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract(doc: Doc):
    p = agent.process("request", doc.text)
    return {"doc_type": p.doc_type, "fields": p.fields, "issues": p.issues, "sources": p.sources}
