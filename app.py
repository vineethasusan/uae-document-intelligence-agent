"""Streamlit app: process a document, review extracted fields and issues, and ask questions across documents."""
import json
from datetime import date

import pandas as pd
import streamlit as st

from docagent import llm
from docagent.agent import DocumentAgent
from docagent.generate import generate

st.set_page_config(page_title="UAE Document Intelligence Agent", layout="wide")
TODAY = date(2026, 10, 1)


@st.cache_resource
def load_agent():
    docs = generate(n=600, seed=7, today=TODAY)
    agent = DocumentAgent(TODAY).fit([d.text for d in docs[:240]], [d.doc_type for d in docs[:240]])
    agent.ingest([(d.doc_id, d.text) for d in docs[240:]])
    return agent, docs[240:]


agent, sample = load_agent()
st.title("UAE Document Intelligence Agent")
st.caption(f"Engine: {'LLM + rules' if llm.llm_available() else 'offline (rules + semantic fallback)'} · Arabic and English")

tab1, tab2, tab3 = st.tabs(["Process a document", "Ask your documents", "Review queue"])

with tab1:
    pick = st.selectbox("Load a sample", [d.doc_id for d in sample[:50]])
    text = st.text_area("Document text (paste OCR output)", next(d.text for d in sample if d.doc_id == pick), height=260)
    if st.button("Process", type="primary"):
        p = agent.process("uploaded", text)
        c1, c2 = st.columns([2, 1])
        c1.subheader(p.doc_type.replace("_", " ").title())
        c1.dataframe(pd.DataFrame([{"field": k, "value": v, "source": p.sources.get(k, "-")} for k, v in p.fields.items()]),
                     hide_index=True, use_container_width=True)
        c2.subheader("Checks")
        if p.issues:
            for i in p.issues:
                c2.error(i.replace("_", " "))
        else:
            c2.success("All checks passed")
        c2.download_button("Download JSON", json.dumps({"type": p.doc_type, "fields": p.fields, "issues": p.issues},
                                                        ensure_ascii=False, indent=2), "document.json")

with tab2:
    q = st.text_input("Question", "What is the VAT amount on invoice " + next(d.fields["invoice_no"] for d in sample if d.doc_type == "tax_invoice") + "?")
    if q:
        a = agent.ask(q)
        st.markdown(f"**Answer:** {a['answer']}")
        st.caption("Sources: " + ", ".join(a["sources"]))

with tab3:
    rows = [{"doc_id": p.doc_id, "type": p.doc_type, "issues": ", ".join(p.issues)} for p in agent.store if p.issues]
    st.metric("Documents needing attention", f"{len(rows)} of {len(agent.store)}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
