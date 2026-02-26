# plmembedder/__init__.py
"""PLMEmbedder - Protein Language Model Embedding Library."""

__version__ = "0.1.0"

from .config import (
    PipelineConfig,
    EmbedderConfig,
    CacheConfig,
    DecoyConfig,
    OutputConfig,
    EmbedderType,
    DecoyType,
)
from .pipeline import EmbeddingPipeline
from .cache import EmbeddingCache
from .decoy.generator import DecoyGenerator, DecoyResult
from .embedders.base import BaseEmbedder, EmbeddingResult
from .embedders import create_embedder
from .io.fasta_parser import FastaParser

__all__ = [
    # Config
    "PipelineConfig",
    "EmbedderConfig",
    "CacheConfig",
    "DecoyConfig",
    "OutputConfig",
    "EmbedderType",
    "DecoyType",
    # Pipeline
    "EmbeddingPipeline",
    # Cache
    "EmbeddingCache",
    # Decoy
    "DecoyGenerator",
    "DecoyResult",
    # Embedders
    "BaseEmbedder",
    "EmbeddingResult",
    "create_embedder",
    # IO
    "FastaParser",
]
