import json
import pickle
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
import math
from nltk.stem import PorterStemmer
from lib.constants import BM25_B, BM25_K1

# --- Paths and settings (relative to the project root you run from) ---
MOVIES_PATH = "data/movies.json"
STOPWORDS_PATH = "data/stopwords.txt"
CACHE_DIR = Path("cache")
INDEX_PATH = CACHE_DIR / "index.pkl"
DOCMAP_PATH = CACHE_DIR / "docmap.pkl"
TF_PATH = CACHE_DIR / "term_frequencies.pkl"
MAX_RESULTS = 5
DOC_LENGTHS_PATH = CACHE_DIR / "doc_lengths.pkl"

# Translation table that deletes every ASCII punctuation character
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

STEMMER = PorterStemmer()  # create once; reused for every token


# --- Text preprocessing ---

def normalize(text: str) -> str:
    """Lowercase and strip punctuation so matching ignores both."""
    return text.lower().translate(PUNCT_TABLE)


def load_stop_words() -> list[str]:
    """Read stop words (one per line) and preprocess them like query/title text."""
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    # normalize() turns "aren't" into "arent"; the `if` skips blank lines
    return [normalize(line) for line in lines if line.strip()]


def tokenize(text: str, stop_words: list[str]) -> list[str]:
    """Normalize, split on whitespace, drop stop words, then stem each token."""
    tokens = [tok for tok in normalize(text).split() if tok not in stop_words]
    # Stem after stop word removal: "running" -> "run", "movies" -> "movi"
    return [STEMMER.stem(tok) for tok in tokens]


def tokenize_term(term: str, stop_words: list[str]) -> str:
    """Tokenize a single term; raise if it doesn't produce exactly one token."""
    tokens = tokenize(term, stop_words)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token from {term!r}, got {tokens}")
    return tokens[0]


# --- Data loading ---

def load_movies() -> list[dict]:
    """Load and return the list of movies from the dataset."""
    with open(MOVIES_PATH, encoding="utf-8") as f:
        return json.load(f)["movies"]


# --- Inverted index ---

class InvertedIndex:
    def __init__(self) -> None:
        # token -> set of doc IDs that contain it
        self.index: dict[str, set[int]] = defaultdict(set)
        # doc ID -> full movie dict
        self.docmap: dict[int, dict] = {}
        # doc ID -> Counter of token -> occurrences in that doc
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        # doc ID -> number of tokens in that doc (after tokenization)
        self.doc_lengths: dict[int, int] = {}
        # Loaded once and reused for every document and query
        self.stop_words = load_stop_words()
        # Cached average doc length; computed lazily, reset whenever the data changes
        self.__avg_doc_length: float | None = None

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize text; update index, term counts, and document length."""
        tokens = tokenize(text, self.stop_words)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        """Average token count per document (cached); 0.0 if the index is empty."""
        if self.__avg_doc_length is None:
            if not self.doc_lengths:
                return 0.0
            self.__avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return self.__avg_doc_length

    def get_documents(self, term: str) -> list[int]:
        """Return sorted doc IDs for a single already-preprocessed token."""
        # .get() so a missing term doesn't insert an empty set into the defaultdict
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        """Return how often an already-preprocessed token appears in a doc (0 if absent)."""
        # .get() avoids creating an empty Counter for an unknown doc_id;
        # a Counter returns 0 for missing keys
        return self.term_frequencies.get(doc_id, Counter())[term]

    def get_idf(self, term: str) -> float:
        """Inverse document frequency for an already-preprocessed token."""
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(term, set()))
        # +1 on both sides avoids division by zero for unseen terms
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        """BM25 IDF for an already-preprocessed token."""
        n = len(self.docmap)                  # total documents
        df = len(self.index.get(term, set()))  # documents containing the term
        # The +1 inside the log keeps the result positive, even for very common terms
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        """Saturated TF, scaled so long documents don't win just by being long."""
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length == 0:
            return 0.0  # empty index: avoid division by zero
        doc_length = self.doc_lengths.get(doc_id, 0)
        # >1 for longer-than-average docs (penalized), <1 for shorter ones (boosted)
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_tfidf(self, doc_id: int, term: str) -> float:
        """TF-IDF for an already-preprocessed token in a given document."""
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def bm25(self, doc_id: int, term: str) -> float:
        """Full BM25 score for an already-preprocessed token in one document."""
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = MAX_RESULTS) -> list[tuple[int, float]]:
        """Score every document against the query; return the top `limit` (doc_id, score)."""
        query_tokens = tokenize(query, self.stop_words)

        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            scores[doc_id] = sum(self.bm25(doc_id, token) for token in query_tokens)

        # Highest score first; sorted() is stable, so ties stay in ascending-ID order
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def build(self) -> None:
        self.__avg_doc_length = None  # data is about to change
        """Index every movie's title and description."""
        for m in load_movies():
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], f"{m['title']} {m['description']}")

    def save(self) -> None:
        """Pickle all index data into the cache directory."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        self.__avg_doc_length = None  # data is about to change
        """Load all cache files; raise if any is missing."""
        for path in (INDEX_PATH, DOCMAP_PATH, TF_PATH, DOC_LENGTHS_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Missing cache file: {path}")
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TF_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(DOC_LENGTHS_PATH, "rb") as f:
            self.doc_lengths = pickle.load(f)


# --- Commands ---

def load_index_or_exit() -> InvertedIndex:
    """Load the cached index, or print an error and exit if it isn't built."""
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}. Run the 'build' command first.", file=sys.stderr)
        sys.exit(1)
    return idx


def build_command() -> None:
    """Build the inverted index and save it to disk."""
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str) -> None:
    """Look up query tokens in the cached index and print up to MAX_RESULTS movies."""
    idx = load_index_or_exit()

    results: list[int] = []
    for token in tokenize(query, idx.stop_words):
        for doc_id in idx.get_documents(token):
            if doc_id not in results:  # a movie can match several query tokens
                results.append(doc_id)
            if len(results) == MAX_RESULTS:
                break
        if len(results) == MAX_RESULTS:
            break

    print(f"Searching for: {query}")
    for i, doc_id in enumerate(results, start=1):
        print(f"{i}. {idx.docmap[doc_id]['title']} (ID: {doc_id})")


def tf_command(doc_id: int, term: str) -> None:
    """Print how many times term appears in the given document."""
    idx = load_index_or_exit()
    token = tokenize_term(term, idx.stop_words)
    print(idx.get_tf(doc_id, token))

def idf_command(term: str) -> None:
    """Print the IDF of a single term across all documents."""
    idx = load_index_or_exit()
    token = tokenize_term(term, idx.stop_words)
    idf = idx.get_idf(token)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tfidf_command(doc_id: int, term: str) -> None:
    """Print the TF-IDF score of a single term in the given document."""
    idx = load_index_or_exit()
    token = tokenize_term(term, idx.stop_words)
    tf_idf = idx.get_tfidf(doc_id, token)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def bm25_idf_command(term: str) -> float:
    """Load the index, tokenize a single term, and return its BM25 IDF."""
    idx = load_index_or_exit()
    token = tokenize_term(term, idx.stop_words)
    return idx.get_bm25_idf(token)

def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    """Load the index, tokenize a single term, and return its BM25 TF in a doc."""
    idx = load_index_or_exit()
    token = tokenize_term(term, idx.stop_words)
    return idx.get_bm25_tf(doc_id, token, k1, b)

def bm25_search_command(query: str, limit: int = MAX_RESULTS) -> list[tuple[int, str, float]]:
    """Load the index and return the top BM25 matches as (doc_id, title, score)."""
    idx = load_index_or_exit()
    return [
        (doc_id, idx.docmap[doc_id]["title"], score)
        for doc_id, score in idx.bm25_search(query, limit)
    ]