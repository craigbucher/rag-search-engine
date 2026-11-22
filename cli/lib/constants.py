"""Shared tuning constants for search scoring."""

BM25_K1 = 1.5  # term-frequency saturation; higher = slower saturation
BM25_B = 0.75  # length normalization strength; 0 = none, 1 = full