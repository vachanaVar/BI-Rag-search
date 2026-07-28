"""
RAG-Based AI Search System — starter interface.

Run with:
    streamlit run app.py
"""

import os
import random
import streamlit as st

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

# Set page config FIRST (must be the first Streamlit command)
st.set_page_config(
    page_title="B.I. Search",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM THEME WITH DARK/LIGHT TOGGLE ============
st.markdown("""
<style>
    /* ============ ONLY BUTTONS ============ */
    /* Primary buttons (Search, Submit) */
    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #FFB300 !important;
        color: #1a1a2e !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(255, 179, 0, 0.2) !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #FFC107 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 193, 7, 0.4) !important;
    }

    .stButton > button:active,
    .stFormSubmitButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ============ RANDOM FACT CARDS ============ */
    .fact-card {
        background-color: #FFF8E1;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FFB300;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .fact-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 16px rgba(255, 179, 0, 0.15);
    }
    
    .fact-number {
        font-weight: 700;
        color: #FFB300;
        font-size: 1.1rem;
        margin-right: 0.5rem;
    }
    
    .fact-source {
        font-size: 0.85rem;
        color: #888;
        margin-left: 0.5rem;
    }
    
    .fact-text {
        margin-top: 0.4rem;
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

DATA_FOLDER = "C:/final_project_starter/data/sample_docs/"
os.environ["GROQ_API_KEY"] = "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"

@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store():
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


store, docs, chunks = load_store()

# ============ FILTER: Get only bee_facts.txt chunks ============
bee_facts_chunks = [chunk for chunk in chunks if "bee_facts" in chunk.source_file.lower()]

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help="Extractive works with no setup. LLM mode needs GROQ_API_KEY set.")
    st.divider()
    st.caption(f"Indexed **{len(docs)}** documents \u2192 **{len(chunks)}** chunks")
    st.caption(f"🐝 **{len(bee_facts_chunks)}** facts available for random selection")
    with st.expander("Documents in this index"):
        for d in docs:
            st.write(f"- {d['title']}")

# ============ MAIN UI ============
st.title("🐝 B.I. Search System")
st.caption("Ask a question about bees below, or click **🎲 Random Fact** for a surprise!")

# ============ TWO COLUMNS: Search Input + Random Button ============
col1, col2 = st.columns([5, 1])

with col1:
    # Search form (with Enter key support)
    with st.form(key="search_form"):
        query = st.text_input(
            "Your question",
            placeholder="e.g. Where can bees be found?",
            label_visibility="collapsed"
        )
        search_clicked = st.form_submit_button("🔍 Search", type="primary")

with col2:
    # Random fact button (outside the form so it works independently)
    random_clicked = st.button("🎲 Random Facts", type="primary", use_container_width=True)

# ============ HANDLE RANDOM BUTTON ============
if random_clicked:
    # ============ ONLY USE BEE_FACTS CHUNKS ============
    if not bee_facts_chunks:
        st.warning("⚠️ No bee facts found! Make sure bee_facts.txt is in your data folder.")
    else:
        # Pick 3 random chunks from bee_facts only
        num_facts = min(3, len(bee_facts_chunks))
        random_chunks = random.sample(bee_facts_chunks, num_facts)

        st.subheader("🐝 Random Bee Facts")

        for i, chunk in enumerate(random_chunks, 1):
            # Extract just the fact number and text (clean up the format)
            fact_text = chunk.text

            # Try to extract the fact number if it's in the text
            import re
            fact_num_match = re.search(r'^(\d+)\.\s*', fact_text)
            fact_num = fact_num_match.group(1) if fact_num_match else str(i)

            # Remove the number prefix if it exists for cleaner display
            if fact_num_match:
                fact_text = fact_text[fact_num_match.end():]

            # Display each fact as a styled card
            st.markdown(f"""
            <div class="fact-card">
                <span>
                    <span class="fact-number">#{fact_num}</span>
                    <span class="fact-source">📄 {chunk.doc_title}</span>
                </span>
                <div class="fact-text">{fact_text[:600]}{"..." if len(fact_text) > 600 else ""}</div>
            </div>
            """, unsafe_allow_html=True)

        # Show source details in expander
        with st.expander("📚 Source details"):
            for chunk in random_chunks:
                st.write(f"- **{chunk.doc_title}** (from `{chunk.source_file}`)")

# ============ HANDLE SEARCH ============
if search_clicked and query.strip():
    with st.spinner("🔎 Searching for an answer..."):
        retrieved = store.query(query, top_k=top_k)
        answer = generate_answer(query, retrieved, mode=mode)

    st.subheader("💡 Answer")
    st.markdown(answer)

    st.subheader("📚 Sources")
    for i, (chunk, score) in enumerate(retrieved, 1):
        with st.expander(f"**{i}. {chunk.doc_title}**  ·  **Similarity: {score:.2f}**"):
            st.write(chunk.text)
elif search_clicked:
    st.warning("Type a question first.")