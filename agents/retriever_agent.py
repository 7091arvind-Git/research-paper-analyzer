from rag.retriever import retrieve_documents
from config import TOP_K

def run_retriever_agent(question, vector_store, client=None, api_key=None, top_k=TOP_K):
    chunks = retrieve_documents(question, vector_store, client=client, api_key=api_key, top_k=top_k)
    return chunks
