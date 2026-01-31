import sys
from pathlib import Path

# ---- force project root into python path ----
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from src.rag_core import load_assets, retrieve, build_evidence_block, generate_with_openai, TOP_K_DEFAULT

st.set_page_config(page_title="IFRS 9 RAG Assistant", layout="wide")
st.title("📄 IFRS 9 / Credit Risk RAG Assistant")
st.caption("FAISS Retrieval + OpenAI Generation with strict evidence grounding and citations.")

# Load retrieval assets once
@st.cache_resource
def cached_assets():
    return load_assets()

try:
    emb_model, index, metas, chunk_text_map = cached_assets()
except Exception as e:
    st.error(f"Failed to load assets: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-k evidence", min_value=3, max_value=10, value=TOP_K_DEFAULT, step=1)

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Ask a question", placeholder="e.g., What is SICR under IFRS 9?")
    run = st.button("Search & Answer")

    if run and query.strip():
        with st.spinner("Retrieving evidence..."):
            evs = retrieve(query, emb_model, index, metas, chunk_text_map, top_k=top_k)
            st.session_state["evs"] = evs

        evidence_block = build_evidence_block(st.session_state["evs"])

        with st.spinner("Generating answer with OpenAI..."):
            try:
                answer = generate_with_openai(query, evidence_block)
                st.session_state["answer"] = answer
                st.session_state["error"] = None
            except Exception as e:
                st.session_state["answer"] = None
                st.session_state["error"] = str(e)

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    if st.session_state.get("answer"):
        st.subheader("🤖 Answer (with citations)")
        st.write(st.session_state["answer"])

with col2:
    st.subheader("🔎 Evidence (Top-k)")
    evs = st.session_state.get("evs", [])
    if not evs:
        st.info("Run a query to see evidence here.")
    else:
        for i, ev in enumerate(evs, 1):
            with st.expander(f"{i}. {ev['source']} p.{ev['page']}"):
                st.caption(ev["chunk_id"])
                st.write(ev["text"][:3000])
