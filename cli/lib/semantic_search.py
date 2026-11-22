from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:
    def __init__(self) -> None:
        # Downloads the model on first use, then loads it from the local cache
        self.model = SentenceTransformer(MODEL_NAME)


def verify_model() -> None:
    """Load the embedding model and print basic info about it."""
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")