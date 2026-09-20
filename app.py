import os
import importlib
import streamlit as st
import config
import rag.embeddings
import rag.pdf_loader
import rag.text_splitter
import agents.analyst_agent
import agents.critic_agent

importlib.reload(config)
importlib.reload(rag.embeddings)
importlib.reload(rag.pdf_loader)
importlib.reload(rag.text_splitter)
importlib.reload(agents.analyst_agent)
importlib.reload(agents.critic_agent)

from config import GOOGLE_API_KEY, UPLOAD_DIR
from rag.pdf_loader import load_pdf
from rag.text_splitter import split_text
from rag.vector_store import create_vector_store
from rag.embeddings import get_client
from pipeline import run_agentic_rag, summarize_paper

st.set_page_config(
    page_title="ResearchMate – Agentic Literature Intelligence",
    page_icon="🔬",
    layout="wide"
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.title("🔬 ResearchMate")
    st.caption("Agentic RAG for Research Paper Analysis")
    st.divider()
    
    st.subheader("Document Ingestion")
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
    process_button = st.button("Index & Analyze Paper", use_container_width=True, type="primary")

    if process_button:
        if not GOOGLE_API_KEY:
            st.error("Gemini API key is not configured. Please add GEMINI_API_KEY to Streamlit Secrets.")
        elif uploaded_file is None:
            st.error("Please upload a PDF file first.")
        else:
            with st.spinner("Extracting text, chunking, and generating FAISS embeddings..."):
                try:
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    pages_data = load_pdf(file_path)
                    if not pages_data:
                        st.error("No text could be extracted from this PDF. It may be empty or scanned.")
                    else:
                        chunks = split_text(pages_data)
                        client = get_client(GOOGLE_API_KEY)
                        vector_store = create_vector_store(chunks, client=client)

                        st.session_state.vector_store = vector_store
                        st.session_state.processed_file_name = uploaded_file.name
                        st.session_state.num_pages = len(pages_data)
                        st.session_state.num_chunks = len(chunks)
                        st.session_state.last_result = None

                        st.success("Paper successfully indexed in FAISS!")
                except Exception as e:
                    st.error(f"Error processing paper: {str(e)}")

    if st.session_state.processed_file_name:
        st.divider()
        st.write(f"**Document:** {st.session_state.processed_file_name}")
        st.write(f"**Pages:** {st.session_state.num_pages}")
        st.write(f"**Indexed Chunks:** {st.session_state.num_chunks}")

st.title("🔬 ResearchMate: Agentic Literature Intelligence")
st.markdown("##### *Agentic RAG for Research Paper Analysis*")
st.markdown("A simple Agentic RAG system that retrieves relevant sections from research papers, analyzes them using Gemini, and verifies the generated answers.")
st.divider()

if st.session_state.vector_store is None:
    st.info("Upload a research paper in the sidebar and click **Index & Analyze Paper** to initialize the agentic pipeline.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Precision Retrieval")
        st.caption("Uses FAISS to find relevant sections from the research paper.")
    with col2:
        st.markdown("### 🧠 Analyst Synthesis")
        st.caption("Gemini analyzes the retrieved content and generates the answer.")
    with col3:
        st.markdown("### 🛡️ Critic Verification")
        st.caption("Critic Agent checks whether the answer is supported by the paper.")
else:
    col_summary, col_spacer = st.columns([1, 2])
    with col_summary:
        summarize_button = st.button("📑 Generate Executive Summary", use_container_width=True)

    user_question = st.text_input(
        "Ask an analytical question about the paper:",
        placeholder="e.g. What baseline methods were compared? What are the main limitations?"
    )
    ask_button = st.button("Run Agentic Analysis", type="primary")

    active_query = None
    if summarize_button:
        active_query = "SUMMARIZE"
    elif ask_button:
        if not user_question.strip():
            st.warning("Please type a question before clicking Run Agentic Analysis.")
        else:
            active_query = user_question.strip()

    if active_query:
        if not GOOGLE_API_KEY:
            st.error("Gemini API key is not configured. Please add GEMINI_API_KEY to Streamlit Secrets.")
        else:
            with st.spinner("Coordinating Retriever, Analyst, and Critic agents..."):
                try:
                    client = get_client(GOOGLE_API_KEY)
                    if active_query == "SUMMARIZE":
                        result = summarize_paper(st.session_state.vector_store, client=client)
                    else:
                        result = run_agentic_rag(active_query, st.session_state.vector_store, client=client)
                    st.session_state.last_result = result
                except Exception as e:
                    st.error(f"Error during execution: {str(e)}")

    if st.session_state.last_result:
        result = st.session_state.last_result

        st.subheader("Agentic RAG Pipeline")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Retriever Agent", f"{len(result['sources'])} Chunks")
        with c2:
            st.metric("Analyst Agent", "Synthesized")
        with c3:
            st.metric("Critic Agent", result["critic_status"])
        with c4:
            st.metric("Self-Correction", "Revised" if result["revised"] else "Verified")

        st.subheader("Synthesized Analysis")
        st.markdown(result["answer"])

        st.subheader("Critic Verification Audit")
        if result["critic_status"] == "SUPPORTED":
            st.success(f"**Verification Status:** {result['critic_status']}\n\n{result['critic_feedback']}")
        else:
            st.warning(f"**Verification Status:** {result['critic_status']}\n\n{result['critic_feedback']}")

        st.subheader("Source Attribution & Evidence")
        for i, src in enumerate(result["sources"], 1):
            page_num = src.get("page", "N/A")
            chunk_id = src.get("chunk_id", i)
            with st.expander(f"Evidence #{i} — Page {page_num} (Chunk {chunk_id})"):
                st.write(src["text"])

