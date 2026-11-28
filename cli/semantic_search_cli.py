import argparse

from lib.semantic_search import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_MAX_CHUNK_SENTENCES,
    chunk_command,
    embed_query_text,
    embed_text,
    semantic_search,
    verify_embeddings,
    verify_model,
    semantic_chunk_command,
    embed_chunks,
)

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Load the embedding model and print its info")

    embed_text_parser = subparsers.add_parser("embed-text", help="Generate an embedding for a text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify-embeddings", help="Build or load movie embeddings and check them")

    embed_query_parser = subparsers.add_parser("embed-query", help="Generate an embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Search query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by meaning")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size word chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP, help="Words shared between consecutive chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic-chunk", help="Split text into chunks on sentence boundaries"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=DEFAULT_MAX_CHUNK_SENTENCES,
        help="Maximum sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP, help="Sentences shared between chunks"
    )

    subparsers.add_parser("embed-chunks", help="Build or load embeddings for description chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed-text":
            embed_text(args.text)
        case "verify-embeddings":
            verify_embeddings()
        case "embed-query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic-chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed-chunks":
            embed_chunks()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()