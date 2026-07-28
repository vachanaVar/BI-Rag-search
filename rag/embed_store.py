"""
Vector store: turn chunks into vectors and support similarity search over them.

This starter ships with a TF-IDF backend (same technique from the Week 14 lab) so
the whole project runs immediately with zero API keys and no model downloads.

Upgrade path (for your final project — do this once the pipeline works end-to-end):
- Swap TfidfVectorizer for real embeddings, e.g.:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts)
- Swap the in-memory cosine_similarity search below for FAISS or Chroma once your
  chunk count grows past a few thousand.
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

from rag.ingest import Chunk


class VectorStore:
    def __init__(self):
        # Simple: just the model and ChromaDB
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./chroma_db/")
        self.collection = self.client.get_or_create_collection(name="chunks")
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Store all chunks as vectors."""
        self.chunks = chunks

        # Clear existing data
        if self.collection.count() > 0:
            self.collection.delete(ids=self.collection.get()['ids'])

        # Get texts
        texts = [c.text for c in chunks]

        # Create embeddings
        embeddings = self.model.encode(texts).tolist()

        # Store in ChromaDB
        self.collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": c.source_file, "title": c.doc_title} for c in chunks]
        )

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return top_k (chunk, score) pairs."""
        if self.collection.count() == 0:
            raise RuntimeError("Call build() first.")

        # Get query embedding
        query_vec = self.model.encode(query_text).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )

        # Return as (Chunk, score) tuples
        output = []
        for i in range(len(results['ids'][0])):
            # Find the original chunk
            chunk = self.chunks[int(results['ids'][0][i].split('_')[1])]
            score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
            output.append((chunk, score))

        return output