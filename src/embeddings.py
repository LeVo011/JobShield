"""Sentence embeddings when cached; TF-IDF local vector fallback otherwise."""
import numpy as np
from src.config import EMBEDDING_MODEL
class Embedder:
    def __init__(self):
        self.backend="tfidf"; self.model=None; self.diagnostic=""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.diagnostic="Sentence Transformers is not installed; using local TF-IDF vectors."
            return
        try:
            self.model=SentenceTransformer(EMBEDDING_MODEL,local_files_only=True); self.backend="sentence-transformers"
        except Exception as exc:
            self.diagnostic=f"Local embedding model unavailable ({type(exc).__name__}); using local TF-IDF vectors."
    def fit_transform(self,texts):
        if self.model is not None: return np.asarray(self.model.encode(texts,normalize_embeddings=True),dtype="float32")
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.model=TfidfVectorizer(ngram_range=(1,2),max_features=50000); return self.model.fit_transform(texts).toarray().astype("float32")
    def transform(self,texts):
        if self.backend=="sentence-transformers": return np.asarray(self.model.encode(texts,normalize_embeddings=True),dtype="float32")
        return self.model.transform(texts).toarray().astype("float32")
