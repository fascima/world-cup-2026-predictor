"""General helper functions."""

from __future__ import annotations

from pathlib import Path


def ensure_directories() -> None:
    """Create project output directories if they do not already exist."""
    for path in [
        Path("data/raw"),
        Path("data/processed"),
        Path("data/fixtures"),
        Path("results"),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_probabilities(probs: list[float]) -> list[float]:
    """Return non-negative probabilities normalized to sum to 1."""
    clipped = [max(0.0, float(prob)) for prob in probs]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped)] * len(clipped)
    return [prob / total for prob in clipped]
