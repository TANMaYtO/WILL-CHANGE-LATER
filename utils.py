import os
import faiss
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
emb_model_name= "all-MiniLM-L6-v2"
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_emb(model_name: str = emb_model_name):
    return SentenceTransformer(model_name)

def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    norms= np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0]=1e-9
    embs = embeddings / norms
    return embs.astype(np.float32)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

def save_mapping(df: pd.DataFrame, path: str):
    df.to_csv(path, index= False)

def load_mapping(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_pickle(obj, path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def tokk_search(index: faiss.Index, query_emb: np.ndarray, k :int=5)-> tuple[np.ndarray, np.ndarray]:
    D,I= index.search(query_emb, k)
    return D, I