from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from lib.keyword_search import load_movies

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path("cache")
EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"


class SemanticSearch:
    def __init__(self) -> None:
        # Downloads the model on first use, then loads it from the local cache
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings: np.ndarray | None = None  # shape (num_docs, 384)
        self.documents: list[dict] | None = None
        self.document_map: dict[int, dict] = {}  # doc ID -> movie dict

    def generate_embedding(self, text: str):
        """Return the embedding vector (a 1-D numpy array) for a single text."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text")
        # encode() takes a list of inputs and returns one embedding per input
        return self.model.encode([text])[0]

    def _set_documents(self, documents: list[dict]) -> None:
        """Store the documents and index them by ID (shared by build and load)."""
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Embed every movie, cache the result to disk, and return it."""
        self._set_documents(documents)
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        # Row i of the result is the embedding for documents[i]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load cached embeddings if they match the documents; otherwise rebuild."""
        self._set_documents(documents)
        if EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(EMBEDDINGS_PATH)
            # Only trust the cache if it has one vector per document
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Return the top `limit` movies by cosine similarity to the query."""
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        # embeddings[i] belongs to documents[i], so zip keeps them paired
        scored = [
            # Why float(...): numpy returns np.float32. Converting it gives you plain 
            # Python floats in the result dicts, which matters later if you serialize 
            # them to JSON
            (float(cosine_similarity(query_embedding, doc_embedding)), doc)
            for doc_embedding, doc in zip(self.embeddings, self.documents)
        ]
        # Sort on the score only; comparing the dicts on a tie would raise TypeError
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in scored[:limit]
        ]


def verify_model() -> None:
    """Load the embedding model and print basic info about it."""
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str) -> None:
    """Embed a single text and print a short summary of the vector."""
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query: str) -> None:
    """Embed a search query and print a short summary of the vector."""
    search = SemanticSearch()
    # generate_embedding already raises ValueError on empty/whitespace input
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings() -> None:
    """Load (or build) embeddings for all movies and print their shape."""
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# Helper:
def semantic_search(query: str, limit: int = 5) -> None:
    """Load embeddings, run a semantic search, and print the results."""
    search = SemanticSearch()
    search.load_or_create_embeddings(load_movies())

    for i, result in enumerate(search.search(query, limit), start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"{result['description'][:100]}...")
        print()