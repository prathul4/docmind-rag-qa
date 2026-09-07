"""
Streamlit UI for DocMind.

Run with:  streamlit run app.py

Streamlit re-runs this whole script top-to-bottom on every user interaction
(every click, every text input). st.session_state is how we keep the loaded
RAG pipeline and chat history alive across those re-runs instead of rebuilding
everything from scratch each time.
"""

import streamlit as st

from rag_chain import DocMindRAG
import config

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")

st.title("📄 DocMind — RAG Document Q&A Assistant")
st.caption("Ask questions about the Northwind Robotics Employee Handbook. "
           "Answers are grounded only in the retrieved document excerpts below.")

# --- Sidebar: index selection + settings ---
with st.sidebar:
    st.header("Settings")
    available_tags = sorted(
        p.name for p in config.INDEX_DIR.iterdir() if p.is_dir()
    ) if config.INDEX_DIR.exists() else []

    if not available_tags:
        st.error("No index found. Run `python src/ingest.py` first.")
        st.stop()

    tag = st.selectbox("Chunking config", available_tags, index=0,
                        help="Which pre-built index to query (different chunk sizes)")
    top_k = st.slider("Top-k chunks retrieved", 1, 8, config.TOP_K)
    threshold = st.slider("Relevance threshold", 0.0, 1.0, config.RELEVANCE_THRESHOLD, 0.05)

    st.divider()
    st.caption("Session stats")
    total_queries = len(st.session_state.get("history", []))
    st.metric("Queries this session", total_queries)
    if total_queries:
        total_tokens = sum(h["tokens_estimated"] for h in st.session_state["history"])
        avg_latency = sum(h["latency_ms"] for h in st.session_state["history"]) / total_queries
        st.metric("Total tokens (est.)", total_tokens)
        st.metric("Avg latency", f"{avg_latency:.0f} ms")


# Rebuild the RAG object only when settings actually change (embedding calls
# are not free/instant, so we don't want to reconnect on every keystroke).
cache_key = (tag, top_k, threshold)
if st.session_state.get("cache_key") != cache_key:
    with st.spinner("Loading index..."):
        st.session_state["rag"] = DocMindRAG(tag=tag, k=top_k, relevance_threshold=threshold)
        st.session_state["cache_key"] = cache_key

if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.text_input("Ask a question about the handbook:",
                          placeholder="e.g. How many days of PTO do I get?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and generating..."):
        result = st.session_state["rag"].answer(question.strip())
    st.session_state["history"].insert(0, result)

for result in st.session_state["history"]:
    st.markdown(f"**Q: {result['question']}**")
    st.markdown(result["answer"])
    meta_cols = st.columns(3)
    meta_cols[0].caption(f"⏱ {result['latency_ms']:.0f} ms")
    meta_cols[1].caption(f"🔢 ~{result['tokens_estimated']} tokens")
    meta_cols[2].caption(f"📎 {len(result['sources'])} source chunk(s)")

    if result["sources"]:
        with st.expander("Show source chunks used for this answer"):
            for i, src in enumerate(result["sources"], start=1):
                st.markdown(f"**Source {i}** — {src.get('source', 'document')}, "
                             f"page {src['page']} — relevance score {src['score']:.3f}")
                st.code(src["text"], language=None)
    st.divider()
