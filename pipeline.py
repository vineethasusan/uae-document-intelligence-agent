"""End-to-end evaluation: train the classifier, process unseen documents, score extraction,
validation and question answering, and write outputs/report.md and outputs/metrics.json."""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from .agent import DocumentAgent, make_questions, norm_answer
from .extract import field_accuracy
from .generate import generate

OUT = Path("outputs")
TODAY = date(2026, 10, 1)
MANUAL_MIN_PER_DOC = 4.0      # assumption: minutes for a person to key in and check one document


def run(n: int = 900, seed: int = 7) -> dict:
    docs = generate(n=n, seed=seed, today=TODAY)
    train, test = docs[: int(n * 0.4)], docs[int(n * 0.4):]
    agent = DocumentAgent(TODAY).fit([d.text for d in train], [d.doc_type for d in train])

    t0 = time.perf_counter()
    processed = agent.ingest([(d.doc_id, d.text) for d in test])
    secs = time.perf_counter() - t0
    truth = {d.doc_id: d for d in test}

    perfect_by_id = {}
    cls_acc = float(np.mean([p.doc_type == truth[p.doc_id].doc_type for p in processed]))
    per_field, doc_perfect = {}, []
    by_style = {"arabic_digits": [], "other": []}
    for p in processed:
        d = truth[p.doc_id]
        if p.doc_type != d.doc_type:
            doc_perfect.append(False)
            continue
        acc = field_accuracy(p.fields, d.fields)
        for k, ok in acc.items():
            per_field.setdefault(f"{d.doc_type}.{k}", []).append(ok)
        doc_perfect.append(all(acc.values()))
        perfect_by_id[p.doc_id] = all(acc.values())
        by_style["arabic_digits" if any(c in d.text for c in "٠١٢٣٤٥٦٧٨٩") else "other"].append(all(acc.values()))
    field_acc = float(np.mean([v for vals in per_field.values() for v in vals]))

    tp = fp = fn = 0
    for p in processed:
        true_set, pred_set = set(truth[p.doc_id].errors), set(p.issues) - {"total_mismatch", "unusual_term", "needs_review"}
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    qs = make_questions(test, np.random.default_rng(seed), n=200)
    hit3 = exact = 0
    for q in qs:
        hits = agent.retrieve(q["question"], k=3)
        hit3 += q["doc_id"] in [h.doc_id for h in hits]
        exact += norm_answer(agent.ask(q["question"])["answer"]) == norm_answer(q["answer"])

    src = {}
    for p in processed:
        for s in p.sources.values():
            src[s] = src.get(s, 0) + 1
    filled = sum(src.values())
    total_fields = sum(len(p.fields) for p in processed)
    review = float(np.mean(["needs_review" in p.issues for p in processed]))
    auto = [perfect_by_id.get(p.doc_id, False) for p in processed if "needs_review" not in p.issues]
    stp_accuracy = float(np.mean(auto))
    issues_count = {}
    for p in processed:
        for i in p.issues:
            issues_count[i] = issues_count.get(i, 0) + 1

    m = {
        "documents_train": len(train), "documents_test": len(test),
        "classification_accuracy": round(cls_acc, 4),
        "field_accuracy": round(field_acc, 4),
        "documents_fully_correct": round(float(np.mean(doc_perfect)), 4),
        "fully_correct_arabic_digit_docs": round(float(np.mean(by_style["arabic_digits"])), 4) if by_style["arabic_digits"] else None,
        "validation_precision": round(precision, 4), "validation_recall": round(recall, 4),
        "qa_questions": len(qs), "qa_retrieval_hit_at_3": round(hit3 / len(qs), 4), "qa_exact_answer": round(exact / len(qs), 4),
        "fields_by_engine": {k: round(v / total_fields, 4) for k, v in src.items()},
        "fields_unfilled_offline": round(1 - filled / total_fields, 4),
        "documents_routed_to_review": round(review, 4),
        "straight_through_accuracy": round(stp_accuracy, 4),
        "seconds_per_document": round(secs / len(test), 5),
        "manual_hours_saved_per_1000_docs": round(1000 * MANUAL_MIN_PER_DOC / 60, 1),
        "issues_flagged": issues_count,
        "field_accuracy_by_field": {k: round(float(np.mean(v)), 3) for k, v in sorted(per_field.items())},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    (OUT / "report.md").write_text(report(m, processed[:3]), encoding="utf-8")
    return m


def report(m: dict, examples) -> str:
    lines = ["# UAE Document Intelligence Agent: evaluation report", "",
             f"Test set: {m['documents_test']} unseen documents (invoices, trade licences, tenancy contracts) in Arabic, English or both.", "",
             "| Metric | Result |", "|---|---|",
             f"| Document type classification | {m['classification_accuracy']:.1%} |",
             f"| Field-level extraction accuracy | {m['field_accuracy']:.1%} |",
             f"| Documents with every field correct | {m['documents_fully_correct']:.1%} |",
             f"| Business-rule checks: precision / recall | {m['validation_precision']:.1%} / {m['validation_recall']:.1%} |",
             f"| Q&A: right document in top 3 | {m['qa_retrieval_hit_at_3']:.1%} |",
             f"| Q&A: exact answer (offline engine) | {m['qa_exact_answer']:.1%} |",
             f"| Fields filled by rules / semantic fallback | {m['fields_by_engine'].get('rules', 0):.1%} / {m['fields_by_engine'].get('semantic', 0):.1%} |",
             f"| Documents routed to LLM or human review (offline) | {m['documents_routed_to_review']:.1%} |",
             f"| Accuracy of documents processed without review | {m['straight_through_accuracy']:.1%} |",
             f"| Processing time per document | {m['seconds_per_document'] * 1000:.1f} ms |", "",
             "## Issues flagged", ""] + [f"- {k}: {v}" for k, v in m["issues_flagged"].items()] + ["", "## Example outputs", ""]
    for p in examples:
        lines += [f"**{p.doc_id}** ({p.doc_type})", "```json", json.dumps({"fields": p.fields, "issues": p.issues}, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
