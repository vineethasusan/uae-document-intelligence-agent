import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TODAY = date(2026, 10, 1)


@pytest.fixture(scope="session")
def docs():
    from docagent.generate import generate
    return generate(n=300, seed=11, today=TODAY)


@pytest.fixture(scope="session")
def agent(docs):
    from docagent.agent import DocumentAgent
    a = DocumentAgent(TODAY).fit([d.text for d in docs[:120]], [d.doc_type for d in docs[:120]])
    a.ingest([(d.doc_id, d.text) for d in docs[120:]])
    return a
