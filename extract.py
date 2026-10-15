"""Document classification and field extraction.

Two extraction engines share one output schema:
  * RuleExtractor - deterministic, bilingual label matching. Fast, free, explainable, works offline.
  * LLMExtractor  - (llm.py) sends the text plus a JSON schema to an LLM; used for unseen layouts.
The agent runs rules first and only calls the LLM for fields the rules could not fill,
which keeps cost low and every value traceable to a line in the document.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .generate import LABELS

AR_TO_EN = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")


def normalize(text: str) -> str:
    """Arabic-Indic digits to Western, collapse OCR spacing, unify separators."""
    t = text.translate(AR_TO_EN)
    t = re.sub(r"[ \t]+", " ", t)
    return "\n".join(line.strip() for line in t.splitlines() if line.strip())


def build_classifier():
    """Character n-grams handle Arabic, English and mixed text without language-specific tokenisers."""
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=4.0))


class RuleExtractor:
    """Find 'label <sep> value' lines using every known English and Arabic label for each field."""

    def __init__(self):
        self.patterns: dict[str, dict[str, re.Pattern]] = {}
        for doc_type, labels in LABELS.items():
            self.patterns[doc_type] = {}
            for key, (en, ar) in labels.items():
                if key == "title":
                    continue
                en, ar = en.translate(AR_TO_EN), ar.translate(AR_TO_EN)
                alts = sorted({re.escape(en), re.escape(ar), re.escape(f"{en} / {ar}")}, key=len, reverse=True)
                self.patterns[doc_type][key] = re.compile(rf"^(?:{'|'.join(alts)})\s*(?::|-)\s*(.+)$", re.M)

    def extract(self, text: str, doc_type: str) -> dict:
        t = normalize(text)
        out = {}
        for key, pat in self.patterns[doc_type].items():
            m = pat.search(t)
            out[key] = m.group(1).strip() if m else None
        return out


DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
AMOUNT_RE = re.compile(r"^[\d,]+\.\d{2}$")
VALUE_KIND = {"invoice_date": "date", "issue_date": "date", "expiry_date": "date", "start_date": "date", "end_date": "date",
              "subtotal": "amount", "vat": "amount", "total": "amount", "annual_rent": "amount",
              "supplier_trn": "digits", "licence_no": "digits", "contract_no": "digits", "cheques": "small_int"}
# extra wording per field, used only for similarity (not exact matching)
DESCRIPTIONS = {"invoice_no": "invoice number reference", "invoice_date": "invoice date tax point",
                "supplier_trn": "tax registration number", "total": "total amount payable",
                "licence_no": "licence number", "expiry_date": "expiry date valid",
                "activity": "licensed activity", "contract_no": "ejari contract number",
                "annual_rent": "rent per year", "cheques": "number of cheques instalments"}


def _kind_ok(kind: str | None, v: str) -> bool:
    if kind == "date":
        return bool(DATE_RE.match(v))
    if kind == "amount":
        return bool(AMOUNT_RE.match(v))
    if kind == "digits":
        return v.replace("-", "").isdigit() and len(v) >= 6
    if kind == "small_int":
        return v.isdigit() and int(v) <= 24
    return not DATE_RE.match(v) and not AMOUNT_RE.match(v)


class SemanticFallback:
    """For fields the rules missed: match unknown labels to fields by character-n-gram similarity,
    then accept only values whose shape fits the field (date, amount, number, text)."""

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        corpus = []
        for labels in LABELS.values():
            for k, (en, ar) in labels.items():
                corpus.append(f"{en} {ar} {DESCRIPTIONS.get(k, '')}")
        self.vec.fit(corpus)

    def fill(self, text: str, doc_type: str, fields: dict) -> dict:
        missing = [k for k, v in fields.items() if v is None]
        if not missing:
            return {}
        found = set(v for v in fields.values() if v)
        lines = [ln for ln in normalize(text).splitlines() if re.search(r"\s(?::|-)\s|:", ln)]
        cands = []
        for ln in lines:
            m = re.match(r"^(.+?)\s*(?::|\s-)\s*(.+)$", ln)
            if not m or m.group(2).strip() in found:
                continue
            cands.append((m.group(1).strip(), m.group(2).strip()))
        if not cands:
            return {}
        lab_m = self.vec.transform([c[0] for c in cands])
        fld_m = self.vec.transform([f"{LABELS[doc_type][k][0]} {LABELS[doc_type][k][1]} {DESCRIPTIONS.get(k, '')}" for k in missing])
        sim = (fld_m @ lab_m.T).toarray()
        out, used = {}, set()
        for i in sim.max(axis=1).argsort()[::-1]:
            k = missing[i]
            for j in sim[i].argsort()[::-1]:
                if j in used or sim[i, j] < self.threshold:
                    break
                if _kind_ok(VALUE_KIND.get(k), cands[j][1]):
                    out[k] = cands[j][1]
                    used.add(j)
                    break
        return out


def field_accuracy(pred: dict, truth: dict) -> dict[str, bool]:
    return {k: (str(pred.get(k) or "").strip() == str(v).strip()) for k, v in truth.items()}
