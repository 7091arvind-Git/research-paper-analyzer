import numpy as np
from rag.embeddings import get_embedding
from config import TOP_K

def retrieve_documents(question, vector_store, client=None, api_key=None, top_k=TOP_K):
    question_embedding = get_embedding(question, client=client, api_key=api_key)
    query_vector = np.array([question_embedding], dtype=np.float32)
    index = vector_store["index"]
    chunks = vector_store["chunks"]
    k = min(top_k, len(chunks))
    distances, indices = index.search(query_vector, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(chunks):
            chunk_data = dict(chunks[idx])
            chunk_data["score"] = float(distances[0][rank])
            results.append(chunk_data)
    return results
