from docagent.agent import make_questions, norm_answer
from docagent.extract import field_accuracy


def test_classifier_and_extraction_quality(docs, agent):
    truth = {d.doc_id: d for d in docs}
    assert sum(p.doc_type == truth[p.doc_id].doc_type for p in agent.store) / len(agent.store) > 0.97
    accs = [all(field_accuracy(p.fields, truth[p.doc_id].fields).values()) for p in agent.store
            if "needs_review" not in p.issues]
    assert sum(accs) / len(accs) > 0.97          # documents that skip review must be right


def test_unknown_labels_are_recovered_by_semantic_fallback(agent):
    text = "TAX INVOICE\nInv. Ref : INV-2026-55555\nSupplier : Al Noor Trading LLC\nTRN : 100000000000001\nSubtotal (AED) : 200.00\nVAT 5% (AED) : 10.00\nAmount Due (AED) : 210.00"
    p = agent.process("X", text)
    assert p.fields["total"] == "210.00" and p.sources["total"] == "semantic"
    assert p.fields["supplier_trn"] == "100000000000001"


def test_question_answering(docs, agent):
    import numpy as np
    qs = make_questions(docs[120:], np.random.default_rng(0), n=40)
    right = sum(norm_answer(agent.ask(q["question"])["answer"]) == norm_answer(q["answer"]) for q in qs)
    assert right / len(qs) > 0.85


def test_no_llm_calls_without_configuration(agent):
    import os

    from docagent import llm
    os.environ.pop("DOCAGENT_LLM", None)
    assert not llm.llm_available()
    assert agent.ask("What is the moon made of?")["engine"] == "extractive"
