import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

GOOGLE_API_KEY = get_api_key()
GEMINI_API_KEY = GOOGLE_API_KEY

EMBEDDING_MODEL = "gemini-embedding-2"
FALLBACK_EMBEDDING_MODELS = [
    "gemini-embedding-2",
    "gemini-embedding-2-preview",
    "gemini-embedding-001"
]
LLM_MODEL = "gemini-flash-latest"
FALLBACK_LLM_MODELS = [
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

UPLOAD_DIR = os.path.join("data", "uploads")
FAISS_INDEX_DIR = "faiss_index"
