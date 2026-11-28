from pathlib import Path
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from lib.keyword_search import load_movies

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path("cache")
EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 0
DEFAULT_MAX_CHUNK_SENTENCES = 4
CHUNK_EMBEDDINGS_PATH = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIR / "chunk_metadata.json"
DOC_CHUNK_SENTENCES = 4  # sentences per chunk when embedding descriptions
DOC_CHUNK_OVERLAP = 1    # sentences shared between neighbouring chunks

# Split on whitespace that comes right after ., ! or ?
# (?<=...) is a lookbehind: it requires the punctuation but doesn't consume it,
# so each sentence keeps its own ending mark.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticSearch:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
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

def _group_with_overlap(items: list[str], size: int, overlap: int, joiner: str) -> list[str]:
    """Group items into windows of `size`, sharing `overlap` items between neighbours."""
    if size < 1:
        raise ValueError("chunk size must be at least 1")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and less than the chunk size")

    step = size - overlap
    chunks = []
    for start in range(0, len(items), step):
        chunks.append(joiner.join(items[start : start + size]))
        # Stop once a window reaches the end, so we never emit a chunk
        # made up entirely of already-covered items
        if start + size >= len(items):
            break
    return chunks

def chunk_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Fixed-size chunking: `chunk_size` words per chunk."""
    return _group_with_overlap(text.split(), chunk_size, overlap, " ")

def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SENTENCES,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Sentence-based chunking: up to `max_chunk_size` sentences per chunk."""
    # strip() plus the filter drop empty pieces from leading or trailing whitespace
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s]
    return _group_with_overlap(sentences, max_chunk_size, overlap, " ")

def chunk_command(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> None:
    """Print the character count, then each numbered word chunk."""
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunk_text(text, chunk_size, overlap), start=1):
        print(f"{i}. {chunk}")

def semantic_chunk_command(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SENTENCES,
    overlap: int = DEFAULT_OVERLAP,
) -> None:
    """Print the character count, then each numbered sentence chunk."""
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(semantic_chunk_text(text, max_chunk_size, overlap), start=1):
        print(f"{i}. {chunk}")

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None  # shape (num_chunks, 384)
        self.chunk_metadata: list[dict] | None = None    # one entry per chunk row

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Chunk every description by sentence, embed all chunks, and cache them."""
        self._set_documents(documents)

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description.strip():
                continue  # nothing to embed

            # Same function the semantic-chunk command uses
            chunks = semantic_chunk_text(description, DOC_CHUNK_SENTENCES, DOC_CHUNK_OVERLAP)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,  # position in self.documents
                        "chunk_idx": chunk_idx,  # position within this movie
                        "total_chunks": len(chunks),
                    }
                )

        # Encode once, after the loop; row i corresponds to chunk_metadata[i]
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load cached chunk embeddings and metadata if both exist; otherwise build them."""
        self._set_documents(documents)
        if CHUNK_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, encoding="utf-8") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)


def embed_chunks() -> None:
    """Load or build chunk embeddings for all movies and report how many there are."""
    documents = load_movies()
    search = ChunkedSemanticSearch()
    embeddings = search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")