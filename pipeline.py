# plmembedder/pipeline.py
"""Main embedding pipeline."""

from typing import Dict, List, Tuple, Optional, Iterator
from pathlib import Path
import logging

from .config import PipelineConfig, DecoyConfig
from .cache import EmbeddingCache
from .decoy.generator import DecoyGenerator, DecoyResult
from .embedders import create_embedder
from .embedders.base import EmbeddingResult
from .io.fasta_parser import FastaParser

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Main pipeline for computing protein embeddings.

    Handles:
    - Loading sequences from FASTA files
    - Generating decoy sequences
    - Computing embeddings with PLMs
    - Caching and retrieving cached embeddings
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the embedding pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()
        self.embedder = None
        self.cache = None

        # Results storage
        self.sequences: List[Tuple[str, str]] = []
        self.decoy_results: List[DecoyResult] = []
        self.embeddings: Dict[str, EmbeddingResult] = {}

        # Initialize cache if enabled
        if self.config.cache.enabled:
            self.cache = EmbeddingCache(
                cache_dir=self.config.cache.cache_dir,
                embedder_config=self.config.embedder
            )

    def load_sequences(self, fasta_path: str) -> List[Tuple[str, str]]:
        """
        Load sequences from a FASTA file.

        Args:
            fasta_path: Path to FASTA file

        Returns:
            List of (sequence_id, sequence) tuples
        """
        self.sequences = FastaParser.parse(
            fasta_path,
            validate=self.config.validate_sequences,
            max_sequences=self.config.max_sequences
        )
        logger.info(f"Loaded {len(self.sequences)} sequences from {fasta_path}")
        return self.sequences

    def generate_decoys(
        self,
        sequences: Optional[List[Tuple[str, str]]] = None
    ) -> List[DecoyResult]:
        """
        Generate decoy sequences.

        Args:
            sequences: Sequences to generate decoys for.
                      Uses loaded sequences if None.

        Returns:
            List of DecoyResult objects
        """
        sequences = sequences or self.sequences
        if not sequences:
            raise ValueError("No sequences loaded. Call load_sequences first.")

        if self.config.decoy.n_decoys == 0:
            return []

        generator = DecoyGenerator(self.config.decoy)
        self.decoy_results = generator.generate_all(sequences)

        logger.info(f"Generated {len(self.decoy_results)} decoy sequences")
        return self.decoy_results

    def get_sequences_for_embedding(self) -> List[Tuple[str, str]]:
        """
        Get all sequences to embed (originals + decoys based on config).

        Returns:
            List of (sequence_id, sequence) tuples
        """
        result = []

        # Add original sequences unless decoys_only
        if not self.config.decoy.decoys_only:
            result.extend(self.sequences)

        # Add decoy sequences
        for decoy in self.decoy_results:
            result.append((decoy.decoy_id, decoy.decoy_sequence))

        return result

    def compute_embeddings(
        self,
        sequences: Optional[List[Tuple[str, str]]] = None,
        use_cache: bool = True
    ) -> Dict[str, EmbeddingResult]:
        """
        Compute embeddings for sequences.

        Args:
            sequences: Sequences to embed. Uses all sequences
                      (originals + decoys) if None.
            use_cache: Whether to use cached embeddings

        Returns:
            Dictionary mapping sequence_id to EmbeddingResult
        """
        if sequences is None:
            sequences = self.get_sequences_for_embedding()

        if not sequences:
            raise ValueError("No sequences to embed.")

        # Initialize embedder if needed
        if self.embedder is None:
            self.embedder = create_embedder(self.config.embedder)
            self.embedder.load_model()

        # Separate cached and uncached sequences
        to_embed = []

        for seq_id, seq in sequences:
            if use_cache and self.cache is not None:
                cached = self.cache.get(seq_id, seq)
                if cached is not None:
                    self.embeddings[seq_id] = cached
                    continue
            to_embed.append((seq_id, seq))

        logger.info(
            f"Found {len(sequences) - len(to_embed)} cached, "
            f"{len(to_embed)} to compute"
        )

        # Compute new embeddings
        if to_embed:
            batch_size = self.config.embedder.batch_size

            for i in range(0, len(to_embed), batch_size):
                batch = to_embed[i:i + batch_size]
                results = self.embedder.embed_batch(batch)

                for result in results:
                    self.embeddings[result.sequence_id] = result

                    # Cache the result
                    if self.cache is not None:
                        self.cache.save(result)

        logger.info(f"Total embeddings: {len(self.embeddings)}")
        return self.embeddings

    def run(
        self,
        fasta_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, EmbeddingResult]:
        """
        Run the full embedding pipeline.

        Args:
            fasta_path: Path to input FASTA file
            output_dir: Output directory (uses config default if None)

        Returns:
            Dictionary of embeddings
        """
        # Step 1: Load sequences
        self.load_sequences(fasta_path)

        # Step 2: Generate decoys (if configured)
        if self.config.decoy.n_decoys > 0:
            self.generate_decoys()

        # Step 3: Compute embeddings
        self.compute_embeddings()

        # Step 4: Save outputs (if configured)
        if output_dir or self.config.output.save_embeddings:
            out_dir = output_dir or self.config.output.output_dir
            self.save_embeddings(out_dir)

        return self.embeddings

    def save_embeddings(self, output_dir: str) -> None:
        """Save embeddings to output directory."""
        import numpy as np

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save as single npz file with all embeddings
        embeddings_dict = {}
        metadata = {}

        for seq_id, result in self.embeddings.items():
            embeddings_dict[seq_id] = result.embeddings
            metadata[seq_id] = {
                'sequence': result.sequence,
                'shape': result.embeddings.shape
            }

        np.savez_compressed(
            out_path / "embeddings.npz",
            **embeddings_dict
        )

        # Save metadata
        import json
        with open(out_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved embeddings to {output_dir}")

    def get_embedding(self, sequence_id: str) -> Optional[EmbeddingResult]:
        """Get embedding for a specific sequence."""
        return self.embeddings.get(sequence_id)

    def iterate_embeddings(self) -> Iterator[Tuple[str, EmbeddingResult]]:
        """Iterate over all computed embeddings."""
        yield from self.embeddings.items()
