"""The document agent: classify -> extract (rules, then LLM for gaps) -> validate -> index for Q&A.

Questions are answered with retrieval-augmented generation over the processed documents:
character n-gram TF-IDF retrieval (robust to Arabic morphology and OCR noise) plus either an
LLM answer with citations or, offline, an extractive answer taken from the retrieved fields.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import llm
from .extract import VALUE_KIND, RuleExtractor, SemanticFallback, build_classifier, normalize
from .generate import LABELS
from .validate import validate

FIELD_SYNONYMS = {
    "vat": ["vat", "tax amount", "ضريبة"], "total": ["total", "amount due", "الإجمالي"], "subtotal": ["subtotal"],
    "supplier_trn": ["trn", "tax registration", "الرقم الضريبي"], "invoice_date": ["invoice date", "dated"],
    "supplier": ["supplier", "vendor", "المورد"], "customer": ["customer", "billed to", "العميل"],
    "expiry_date": ["expire", "expiry", "valid until", "تاريخ الانتهاء"], "issue_date": ["issued", "issue date"],
    "activity": ["activity", "business activity", "النشاط"], "authority": ["authority", "issued by"],
    "manager": ["manager", "المدير"], "licence_no": ["licence number", "license number"],
    "annual_rent": ["rent", "annual rent", "الإيجار"], "cheques": ["cheques", "checks", "الشيكات"],
    "tenant": ["tenant", "المستأجر"], "landlord": ["landlord", "owner", "المؤجر"], "property": ["property", "unit"],
    "end_date": ["contract end", "lease end", "ends"], "start_date": ["contract start", "lease start", "starts"],
}


@dataclass
class Processed:
    doc_id: str
    doc_type: str
    fields: dict
    issues: list
    text: str
    sources: dict = field(default_factory=dict)   # field -> "rules" | "llm"


class DocumentAgent:
    def __init__(self, today: date):
        self.today = today
        self.classifier = build_classifier()
        self.rules = RuleExtractor()
        self.fallback = SemanticFallback()
        self.store: list[Processed] = []
        self._vec = None
        self._mat = None

    def fit(self, texts: list[str], types: list[str]) -> "DocumentAgent":
        self.classifier.fit([normalize(t) for t in texts], types)
        return self

    def process(self, doc_id: str, text: str) -> Processed:
        doc_type = self.classifier.predict([normalize(text)])[0]
        fields = self.rules.extract(text, doc_type)
        sources = {k: "rules" for k, v in fields.items() if v is not None}
        for k, v in self.fallback.fill(text, doc_type, fields).items():
            fields[k], sources[k] = v, "semantic"
        missing = [k for k, v in fields.items() if v is None]
        if missing and llm.llm_available():
            got = llm.llm_extract(text, doc_type)
            for k in missing:
                if got.get(k) is not None:
                    fields[k], sources[k] = str(got[k]), "llm"
        issues = validate(doc_type, fields, self.today)
        if any(v is None for v in fields.values()):
            issues.append("needs_review")          # never auto-approve a document with gaps
        elif any(VALUE_KIND.get(k) in {"date", "amount", "digits", "small_int"} and re.search(r"[A-Za-z]", str(v))
                 for k, v in fields.items()):
            issues.append("needs_review")          # likely OCR misread (letter inside a number)
        return Processed(doc_id, doc_type, fields, issues, text, sources)

    def ingest(self, items: list[tuple[str, str]]) -> list[Processed]:
        self.store = [self.process(i, t) for i, t in items]
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._mat = self._vec.fit_transform([self._passage(p) for p in self.store])
        return self.store

    @staticmethod
    def _passage(p: Processed) -> str:
        kv = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in p.fields.items() if v)
        return f"{p.doc_type.replace('_', ' ')} {p.doc_id}. {kv}"

    def retrieve(self, question: str, k: int = 3) -> list[Processed]:
        q = self._vec.transform([normalize(question)])
        scores = (self._mat @ q.T).toarray().ravel()
        return [self.store[i] for i in np.argsort(-scores)[:k]]

    def _target_field(self, question: str, doc_type: str) -> str | None:
        ql = question.lower()
        best, best_len = None, 0
        for f, syns in FIELD_SYNONYMS.items():
            if f not in LABELS[doc_type]:
                continue
            for s in syns:
                if s in ql and len(s) > best_len:
                    best, best_len = f, len(s)
        return best

    def ask(self, question: str) -> dict:
        hits = self.retrieve(question)
        if llm.llm_available():
            text = llm.complete(llm.answer_prompt(question, [self._passage(h) for h in hits]), system="You are a precise assistant.")
            return {"answer": text, "sources": [h.doc_id for h in hits], "engine": "llm"}
        top = hits[0]
        f = self._target_field(question, top.doc_type)
        if f is None or not top.fields.get(f):
            return {"answer": "I could not find that in the documents.", "sources": [h.doc_id for h in hits], "engine": "extractive"}
        return {"answer": top.fields[f], "field": f, "sources": [top.doc_id], "engine": "extractive"}


def make_questions(docs, rng: np.random.Generator, n: int = 150) -> list[dict]:
    """Questions a user would ask, each naming a unique identifier from one document."""
    key_of = {"tax_invoice": "invoice_no", "trade_licence": "company", "tenancy_contract": "contract_no"}
    asks = {"tax_invoice": [("What is the VAT amount on invoice {k}?", "vat"), ("Who is the supplier on invoice {k}?", "supplier"),
                            ("What is the total of invoice {k}?", "total")],
            "trade_licence": [("When does the licence of {k} expire?", "expiry_date"), ("What is the business activity of {k}?", "activity")],
            "tenancy_contract": [("What is the annual rent under Ejari {k}?", "annual_rent"),
                                 ("How many cheques for Ejari contract {k}?", "cheques"), ("Who is the tenant on Ejari {k}?", "tenant")]}
    seen_company = {}
    for d in docs:
        if d.doc_type == "trade_licence":
            seen_company[d.fields["company"]] = seen_company.get(d.fields["company"], 0) + 1
    out = []
    for i in rng.permutation(len(docs)):
        d = docs[i]
        if d.doc_type == "trade_licence" and seen_company[d.fields["company"]] > 1:
            continue                                         # skip ambiguous company names
        tpl, f = asks[d.doc_type][rng.integers(len(asks[d.doc_type]))]
        out.append({"question": tpl.format(k=d.fields[key_of[d.doc_type]]), "doc_id": d.doc_id, "answer": d.fields[f]})
        if len(out) >= n:
            break
    return out


def norm_answer(a) -> str:
    return re.sub(r"\s+", " ", str(a or "")).strip().lower()
