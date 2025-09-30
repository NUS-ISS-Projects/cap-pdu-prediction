"""Vector store module for semantic search over PDU data."""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple


class VectorStore:
    """Vector store for semantic search using sentence transformers."""

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.texts = []

    def build_index(self, texts: List[str]):
        """Build FAISS index from text chunks."""
        if not texts:
            print("No texts provided for vector store")
            return

        self.texts = texts
        print(f"Building vector store with {len(texts)} chunks...")

        embeddings = self.model.encode(texts)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype("float32"))

        print(f"Vector store built with {len(self.texts)} vectors.")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant text chunks using semantic similarity."""
        if not self.index or not self.texts:
            return []

        query_embedding = self.model.encode([query])
        normalized_query = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        scores, indices = self.index.search(
            normalized_query.astype("float32"), min(top_k, len(self.texts))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(score)))

        return results
