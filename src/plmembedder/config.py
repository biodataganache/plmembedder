# plmembedder/config.py
"""Configuration for PLMEmbedder."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from pathlib import Path


class EmbedderType(Enum):
    """Supported protein language models."""
    ESM2 = "esm2"
    ESM1B = "esm1b"
    PROTBERT = "protbert"
    PROTT5 = "prott5"


class DecoyType(Enum):
    """Types of decoy generation methods."""
    REVERSED = "reversed"
    PERMUTED = "permuted"
    SHUFFLED_BLOCKS = "shuffled_blocks"


@dataclass
class EmbedderConfig:
    """Configuration for the embedder."""
    embedder_type: EmbedderType = EmbedderType.ESM2
    model_name: str = "esm2_t33_650M_UR50D"
    device: str = "cuda"
    layer: int = -1
    batch_size: int = 4
    max_sequence_length: int = 1024


@dataclass
class CacheConfig:
    """Configuration for embedding caching."""
    enabled: bool = False
    cache_dir: str = "embeddings_cache"

    # Cache key components
    include_model_hash: bool = True
    include_sequence_hash: bool = True


@dataclass
class DecoyConfig:
    """Configuration for decoy generation."""
    n_decoys: int = 0
    decoy_type: DecoyType = DecoyType.REVERSED
    decoy_prefix: str = "DECOY_"
    random_seed: Optional[int] = None
    decoys_only: bool = False


@dataclass
class OutputConfig:
    """Configuration for output."""
    output_dir: str = "output"
    save_embeddings: bool = True
    save_format: str = "npz"  # npz, hdf5 (future)


@dataclass
class PipelineConfig:
    """Main configuration for the embedding pipeline."""
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    decoy: DecoyConfig = field(default_factory=DecoyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Input options
    max_sequences: Optional[int] = None
    validate_sequences: bool = True
