import os
import json
import numpy as np
import faiss
from rag.embeddings import get_embeddings_batch
from config import FAISS_INDEX_DIR

def create_vector_store(chunks, client=None, api_key=None):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings_batch(texts, client=client, api_key=api_key)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return {
        "index": index,
        "chunks": chunks
    }

def save_vector_store(vector_store, folder_path=FAISS_INDEX_DIR):
    os.makedirs(folder_path, exist_ok=True)
    index_file = os.path.join(folder_path, "index.faiss")
    chunks_file = os.path.join(folder_path, "chunks.json")
    faiss.write_index(vector_store["index"], index_file)
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(vector_store["chunks"], f, ensure_ascii=False, indent=2)

def load_vector_store(folder_path=FAISS_INDEX_DIR):
    index_file = os.path.join(folder_path, "index.faiss")
    chunks_file = os.path.join(folder_path, "chunks.json")
    if not os.path.exists(index_file) or not os.path.exists(chunks_file):
        return None
    index = faiss.read_index(index_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {
        "index": index,
        "chunks": chunks
    }
