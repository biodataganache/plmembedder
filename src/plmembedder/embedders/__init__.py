# plmembedder/embedders/__init__.py
"""Embedder factory and exports."""

from typing import Union

from .base import BaseEmbedder, EmbeddingResult
from .esm_embedder import ESMEmbedder, ESM2Embedder
from .protbert_embedder import ProtBertEmbedder
from .prott5_embedder import ProtT5Embedder, ProtT5XLEmbedder
from ..config import EmbedderConfig, EmbedderType


def create_embedder(config: Union[EmbedderConfig, dict]) -> BaseEmbedder:
    """
    Factory function to create embedder based on configuration.

    Args:
        config: EmbedderConfig object or dict with embedder settings

    Returns:
        Initialized embedder instance
    """
    if isinstance(config, dict):
        config = EmbedderConfig(**config)

    embedder_map = {
        EmbedderType.ESM2: ESMEmbedder,
        EmbedderType.ESM1B: ESMEmbedder,
        EmbedderType.PROTBERT: ProtBertEmbedder,
        EmbedderType.PROTT5: ProtT5Embedder,
    }

    embedder_class = embedder_map.get(config.embedder_type)
    if embedder_class is None:
        raise ValueError(f"Unknown embedder type: {config.embedder_type}")

    return embedder_class(
        model_name=config.model_name,
        device=config.device,
        layer=config.layer
    )


__all__ = [
    "BaseEmbedder",
    "EmbeddingResult",
    "ESMEmbedder",
    "ESM2Embedder",
    "ProtBertEmbedder",
    "ProtT5Embedder",
    "ProtT5XLEmbedder",
    "create_embedder",
]
