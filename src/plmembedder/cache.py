# src/plmembedder/cache.py
"""Embedding cache management for plmembedder.

This module provides caching functionality for protein embeddings,
allowing re-use of computed embeddings across pipeline runs with
different parameters (e.g., different k-mer sizes).

Cache Format:
- Currently uses NumPy's .npz format for simplicity and compatibility
- Each sequence is stored in a separate file for incremental saving
- A manifest.json tracks cached sequences and metadata

Future Improvements:
- HDF5 support for very large datasets (better compression, partial reads)
- Zarr support for cloud storage compatibility
- LRU eviction policy for cache size management
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from .config import EmbedderConfig
from .embedders.base import EmbeddingResult

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for embedding caching."""
    # Whether to cache protein embeddings
    enabled: bool = False

    # Directory for cache storage (default: output_dir/.cache)
    cache_dir: Optional[str] = None

    # Cache format: 'npz' for numpy compressed
    # Future options could include 'h5' for HDF5 (better for very large datasets)
    # or 'zarr' for cloud-compatible storage
    cache_format: str = "npz"

    # Whether to save embeddings incrementally (after each sequence)
    # vs. at the end of batch processing
    # Incremental is safer (preserves progress on crash) but may be slower
    incremental_save: bool = True

    # Whether to verify cache integrity on load
    verify_on_load: bool = True

    # Maximum cache size in GB (None for unlimited)
    # If exceeded, oldest entries are removed
    max_cache_size_gb: Optional[float] = None


@dataclass
class CacheEntry:
    """Metadata for a cached embedding."""
    sequence_id: str
    sequence_hash: str
    model_name: str
    layer: int
    embedding_dim: int
    sequence_length: int
    cached_at: str
    file_path: str


@dataclass
class CacheManifest:
    """Manifest tracking all cached embeddings."""
    version: str = "1.1"
    model_name: str = ""
    layer: int = -1
    embedding_dim: int = 0
    max_sequence_length: int = 1024  # Track for truncation consistency
    entries: Dict[str, CacheEntry] = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if self.entries is None:
            self.entries = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class EmbeddingCache:
    """
    Cache manager for protein embeddings.

    Stores per-residue embeddings for protein sequences, allowing them
    to be reused across pipeline runs with different k-mer parameters.

    Storage Structure:
        cache_dir/
        ├── manifest.json           # Tracks all cached sequences
        └── embeddings/
            ├── {seq_hash_1}.npz    # Embedding for sequence 1
            ├── {seq_hash_2}.npz    # Embedding for sequence 2
            └── ...

    The cache key is based on:
    - Model name (e.g., esm2_t33_650M_UR50D)
    - Layer used for embeddings
    - Max sequence length (for truncation consistency)
    - Sequence content hash (MD5 of effective/truncated sequence)

    This ensures that embeddings are only reused when the exact same
    model configuration was used.
    """

    def __init__(
        self,
        cache_dir: str,
        config: CacheConfig,
        embedder_config: EmbedderConfig
    ):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory for cache storage
            config: Cache configuration
            embedder_config: Embedder configuration (for cache key generation)
        """
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.embedder_config = embedder_config

        # Store max_sequence_length for truncation consistency
        self.max_sequence_length = embedder_config.max_sequence_length

        self.embeddings_dir = self.cache_dir / "embeddings"
        self.manifest_path = self.cache_dir / "manifest.json"

        self.manifest: Optional[CacheManifest] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize cache directory and load manifest."""
        if self._initialized:
            return

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Load or create manifest
        if self.manifest_path.exists():
            self._load_manifest()

            # Check if manifest matches current configuration
            if not self._validate_manifest():
                logger.warning("Cache manifest doesn't match current config, creating new cache")
                self._create_new_manifest()
        else:
            self._create_new_manifest()

        self._initialized = True
        logger.info(f"Embedding cache initialized at {self.cache_dir}")
        logger.info(f"Cache contains {len(self.manifest.entries)} sequences")

    def _load_manifest(self) -> None:
        """Load manifest from disk."""
        try:
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)

            # Reconstruct CacheEntry objects
            entries = {}
            for seq_id, entry_data in data.get('entries', {}).items():
                entries[seq_id] = CacheEntry(**entry_data)

            self.manifest = CacheManifest(
                version=data.get('version', '1.0'),
                model_name=data.get('model_name', ''),
                layer=data.get('layer', -1),
                embedding_dim=data.get('embedding_dim', 0),
                max_sequence_length=data.get('max_sequence_length', 1024),
                entries=entries,
                created_at=data.get('created_at', ''),
                updated_at=data.get('updated_at', '')
            )
        except Exception as e:
            logger.warning(f"Failed to load cache manifest: {e}")
            self._create_new_manifest()

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        self.manifest.updated_at = datetime.now().isoformat()

        # Convert to serializable format
        data = {
            'version': self.manifest.version,
            'model_name': self.manifest.model_name,
            'layer': self.manifest.layer,
            'embedding_dim': self.manifest.embedding_dim,
            'max_sequence_length': self.manifest.max_sequence_length,
            'entries': {
                seq_id: asdict(entry)
                for seq_id, entry in self.manifest.entries.items()
            },
            'created_at': self.manifest.created_at,
            'updated_at': self.manifest.updated_at
        }

        with open(self.manifest_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _create_new_manifest(self) -> None:
        """Create a new manifest."""
        self.manifest = CacheManifest(
            model_name=self.embedder_config.model_name,
            layer=self.embedder_config.layer,
            embedding_dim=0,  # Will be set when first embedding is cached
            max_sequence_length=self.max_sequence_length,
        )
        self._save_manifest()

    def _validate_manifest(self) -> bool:
        """Check if manifest matches current embedder configuration."""
        if self.manifest is None:
            return False

        # Check model name
        if self.manifest.model_name != self.embedder_config.model_name:
            logger.warning(f"Model mismatch: cache has {self.manifest.model_name}, "
                         f"current is {self.embedder_config.model_name}")
            return False

        # Check layer
        if self.manifest.layer != self.embedder_config.layer:
            logger.warning(f"Layer mismatch: cache has {self.manifest.layer}, "
                         f"current is {self.embedder_config.layer}")
            return False

        # Check max_sequence_length - important for truncation consistency
        if self.manifest.max_sequence_length != self.max_sequence_length:
            logger.warning(f"Max sequence length mismatch: cache has {self.manifest.max_sequence_length}, "
                         f"current is {self.max_sequence_length}")
            return False

        return True

    def _get_effective_sequence(self, sequence: str) -> str:
        """
        Get the effective sequence that will be embedded (after truncation).

        This ensures cache keys match what actually gets embedded.

        Args:
            sequence: Original amino acid sequence

        Returns:
            Truncated sequence if longer than max_sequence_length, otherwise original
        """
        if len(sequence) > self.max_sequence_length:
            return sequence[:self.max_sequence_length]
        return sequence

    @staticmethod
    def _compute_sequence_hash(sequence: str) -> str:
        """Compute hash for a sequence."""
        return hashlib.md5(sequence.encode()).hexdigest()

    def _get_cache_path(self, sequence_hash: str) -> Path:
        """Get path for cached embedding file."""
        return self.embeddings_dir / f"{sequence_hash}.npz"

    def has_cached(self, sequence_id: str, sequence: str) -> bool:
        """
        Check if embedding is cached for a sequence.

        Args:
            sequence_id: Sequence identifier
            sequence: Amino acid sequence (original, before truncation)

        Returns:
            True if cached embedding exists and is valid
        """
        if not self._initialized:
            self.initialize()

        # Use effective (potentially truncated) sequence for hash
        effective_seq = self._get_effective_sequence(sequence)
        seq_hash = self._compute_sequence_hash(effective_seq)

        # Check manifest
        if sequence_id not in self.manifest.entries:
            return False

        entry = self.manifest.entries[sequence_id]

        # Verify hash matches (sequence content hasn't changed)
        if entry.sequence_hash != seq_hash:
            return False

        # Verify file exists
        cache_path = self._get_cache_path(seq_hash)
        if not cache_path.exists():
            return False

        return True

    def get_cached_ids(self, sequences: List[Tuple[str, str]]) -> Tuple[Set[str], Set[str]]:
        """
        Partition sequences into cached and uncached.

        Args:
            sequences: List of (sequence_id, sequence) tuples

        Returns:
            Tuple of (cached_ids, uncached_ids)
        """
        if not self._initialized:
            self.initialize()

        cached_ids = set()
        uncached_ids = set()

        for seq_id, seq in sequences:
            if self.has_cached(seq_id, seq):
                cached_ids.add(seq_id)
            else:
                uncached_ids.add(seq_id)

        return cached_ids, uncached_ids

    def load(self, sequence_id: str, sequence: str) -> Optional[EmbeddingResult]:
        """
        Load cached embedding for a sequence.

        Args:
            sequence_id: Sequence identifier
            sequence: Amino acid sequence (original, before truncation)

        Returns:
            EmbeddingResult if cached, None otherwise
        """
        if not self._initialized:
            self.initialize()

        if not self.has_cached(sequence_id, sequence):
            return None

        entry = self.manifest.entries[sequence_id]
        cache_path = self._get_cache_path(entry.sequence_hash)

        try:
            data = np.load(cache_path)
            embeddings = data['embeddings']

            # Get effective sequence for validation
            effective_seq = self._get_effective_sequence(sequence)

            # Verify dimensions against effective (truncated) sequence
            if self.config.verify_on_load:
                if embeddings.shape[0] != len(effective_seq):
                    logger.warning(f"Cached embedding length mismatch for {sequence_id}: "
                                 f"expected {len(effective_seq)}, got {embeddings.shape[0]}")
                    return None

            return EmbeddingResult(
                sequence_id=sequence_id,
                sequence=effective_seq,  # Return effective sequence
                embeddings=embeddings
            )

        except Exception as e:
            logger.warning(f"Failed to load cached embedding for {sequence_id}: {e}")
            return None

    def save(self, result: EmbeddingResult) -> None:
        """
        Save embedding to cache.

        Args:
            result: EmbeddingResult to cache (sequence should already be truncated if needed)
        """
        if not self._initialized:
            self.initialize()

        # The result.sequence should already be the effective (truncated) sequence
        # from the embedder, so we hash it directly
        seq_hash = self._compute_sequence_hash(result.sequence)
        cache_path = self._get_cache_path(seq_hash)

        try:
            # Save embedding
            # Note: For future HDF5 support, this could use h5py instead
            # h5py would allow partial reads and better compression for large datasets
            np.savez_compressed(
                cache_path,
                embeddings=result.embeddings,
                # Could add attention weights here if needed
            )

            # Update manifest
            entry = CacheEntry(
                sequence_id=result.sequence_id,
                sequence_hash=seq_hash,
                model_name=self.embedder_config.model_name,
                layer=self.embedder_config.layer,
                embedding_dim=result.embeddings.shape[1],
                sequence_length=len(result.sequence),
                cached_at=datetime.now().isoformat(),
                file_path=str(cache_path.relative_to(self.cache_dir))
            )

            self.manifest.entries[result.sequence_id] = entry

            # Update embedding dim if not set
            if self.manifest.embedding_dim == 0:
                self.manifest.embedding_dim = result.embeddings.shape[1]

            # Save manifest (if incremental saving is enabled)
            if self.config.incremental_save:
                self._save_manifest()

        except Exception as e:
            logger.error(f"Failed to cache embedding for {result.sequence_id}: {e}")

    def save_batch(self, results: List[EmbeddingResult]) -> None:
        """
        Save multiple embeddings to cache.

        Args:
            results: List of EmbeddingResult objects to cache
        """
        for result in results:
            self.save(result)

        # Save manifest once at end if not doing incremental saves
        if not self.config.incremental_save:
            self._save_manifest()

    def load_batch(
        self,
        sequences: List[Tuple[str, str]]
    ) -> Tuple[List[EmbeddingResult], List[Tuple[str, str]]]:
        """
        Load cached embeddings for a batch of sequences.

        Args:
            sequences: List of (sequence_id, sequence) tuples

        Returns:
            Tuple of (cached_results, uncached_sequences)
        """
        cached_results = []
        uncached_sequences = []

        for seq_id, seq in sequences:
            result = self.load(seq_id, seq)
            if result is not None:
                cached_results.append(result)
            else:
                uncached_sequences.append((seq_id, seq))

        logger.info(f"Loaded {len(cached_results)} cached embeddings, "
                   f"{len(uncached_sequences)} need computation")

        return cached_results, uncached_sequences

    def clear(self) -> None:
        """Clear all cached embeddings."""
        import shutil

        if self.embeddings_dir.exists():
            shutil.rmtree(self.embeddings_dir)

        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._create_new_manifest()

        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if not self._initialized:
            self.initialize()

        total_size = sum(
            f.stat().st_size for f in self.embeddings_dir.glob("*.npz")
        )

        return {
            'num_cached_sequences': len(self.manifest.entries),
            'cache_size_mb': total_size / (1024 * 1024),
            'model_name': self.manifest.model_name,
            'layer': self.manifest.layer,
            'embedding_dim': self.manifest.embedding_dim,
            'max_sequence_length': self.manifest.max_sequence_length,
            'cache_dir': str(self.cache_dir)
        }

    def remove_entry(self, sequence_id: str) -> bool:
        """
        Remove a specific entry from the cache.

        Args:
            sequence_id: Sequence identifier to remove

        Returns:
            True if entry was removed, False if not found
        """
        if not self._initialized:
            self.initialize()

        if sequence_id not in self.manifest.entries:
            return False

        entry = self.manifest.entries[sequence_id]
        cache_path = self._get_cache_path(entry.sequence_hash)

        # Remove file
        if cache_path.exists():
            cache_path.unlink()

        # Remove from manifest
        del self.manifest.entries[sequence_id]
        self._save_manifest()

        logger.info(f"Removed cache entry for {sequence_id}")
        return True

    def list_cached_sequences(self) -> List[str]:
        """
        List all cached sequence IDs.

        Returns:
            List of sequence IDs in cache
        """
        if not self._initialized:
            self.initialize()

        return list(self.manifest.entries.keys())
