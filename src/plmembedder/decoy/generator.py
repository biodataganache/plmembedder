# src/snaikmer/decoy/generator.py
"""Decoy sequence generation from protein sequences."""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
import random

logger = logging.getLogger(__name__)


class DecoyType(Enum):
    """Types of decoy generation methods."""
    REVERSED = "reversed"
    PERMUTED = "permuted"
    SHUFFLED_BLOCKS = "shuffled_blocks"


@dataclass
class DecoyConfig:
    """Configuration for decoy generation."""

    # Decoy generation method
    decoy_type: DecoyType = DecoyType.REVERSED

    # Number of decoys to generate per input sequence
    n_decoys: int = 0

    # Prefix tag for decoy sequence IDs
    decoy_prefix: str = "DECOY_"

    # Whether to include original sequences in output
    include_original: bool = True

    # Random seed for reproducibility (for permuted/shuffled)
    random_seed: Optional[int] = None

    # Block size for shuffled_blocks method
    block_size: int = 5

    # Whether to generate only decoys (no original sequences)
    decoys_only: bool = False


@dataclass
class DecoyResult:
    """Result container for decoy generation."""
    original_id: str
    original_sequence: str
    decoy_id: str
    decoy_sequence: str
    decoy_type: DecoyType
    decoy_index: int  # Which decoy number (0-indexed) for this original


class DecoyGenerator:
    """Generate decoy sequences from protein sequences."""

    def __init__(self, config: Optional[DecoyConfig] = None):
        """
        Initialize decoy generator.

        Args:
            config: DecoyConfig object with generation settings
        """
        self.config = config or DecoyConfig()

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    def generate_decoy(
        self,
        sequence: str,
        decoy_type: Optional[DecoyType] = None,
        decoy_index: int = 0
    ) -> str:
        """
        Generate a single decoy sequence.

        Args:
            sequence: Original amino acid sequence
            decoy_type: Type of decoy to generate (uses config default if None)
            decoy_index: Index of decoy (used for seed variation in random methods)

        Returns:
            Decoy sequence string
        """
        dtype = decoy_type or self.config.decoy_type

        if dtype == DecoyType.REVERSED:
            return self._generate_reversed(sequence)
        elif dtype == DecoyType.PERMUTED:
            return self._generate_permuted(sequence, decoy_index)
        elif dtype == DecoyType.SHUFFLED_BLOCKS:
            return self._generate_shuffled_blocks(sequence, decoy_index)
        else:
            raise ValueError(f"Unknown decoy type: {dtype}")

    def _generate_reversed(self, sequence: str) -> str:
        """Generate reversed sequence."""
        return sequence[::-1]

    def _generate_permuted(self, sequence: str, decoy_index: int = 0) -> str:
        """Generate randomly permuted sequence."""
        # Use decoy_index to create variation between multiple decoys
        if self.config.random_seed is not None:
            local_seed = self.config.random_seed + decoy_index + hash(sequence) % 10000
            rng = np.random.RandomState(local_seed)
        else:
            rng = np.random.RandomState()

        seq_list = list(sequence)
        rng.shuffle(seq_list)
        return ''.join(seq_list)

    def _generate_shuffled_blocks(self, sequence: str, decoy_index: int = 0) -> str:
        """Generate sequence with shuffled blocks (preserves local structure)."""
        block_size = self.config.block_size

        # Split into blocks
        blocks = [
            sequence[i:i + block_size]
            for i in range(0, len(sequence), block_size)
        ]

        # Shuffle blocks
        if self.config.random_seed is not None:
            local_seed = self.config.random_seed + decoy_index + hash(sequence) % 10000
            rng = np.random.RandomState(local_seed)
        else:
            rng = np.random.RandomState()

        rng.shuffle(blocks)
        return ''.join(blocks)

    def generate_decoy_id(
        self,
        original_id: str,
        decoy_type: DecoyType,
        decoy_index: int = 0
    ) -> str:
        """
        Generate decoy sequence ID with appropriate prefix.

        Args:
            original_id: Original sequence ID
            decoy_type: Type of decoy
            decoy_index: Index for multiple decoys per sequence

        Returns:
            Decoy sequence ID with prefix tag
        """
        prefix = self.config.decoy_prefix
        type_tag = decoy_type.value[0].upper()  # R for reversed, P for permuted, S for shuffled

        if self.config.n_decoys > 1:
            return f"{prefix}{type_tag}{decoy_index}_{original_id}"
        else:
            return f"{prefix}{type_tag}_{original_id}"

    def generate_decoys_for_sequence(
        self,
        sequence_id: str,
        sequence: str,
        n_decoys: Optional[int] = None,
        decoy_type: Optional[DecoyType] = None
    ) -> List[DecoyResult]:
        """
        Generate all decoys for a single sequence.

        Args:
            sequence_id: Original sequence ID
            sequence: Original sequence
            n_decoys: Number of decoys (uses config default if None)
            decoy_type: Type of decoy (uses config default if None)

        Returns:
            List of DecoyResult objects
        """
        n = n_decoys if n_decoys is not None else self.config.n_decoys
        dtype = decoy_type or self.config.decoy_type

        results = []
        for i in range(n):
            decoy_seq = self.generate_decoy(sequence, dtype, i)
            decoy_id = self.generate_decoy_id(sequence_id, dtype, i)

            results.append(DecoyResult(
                original_id=sequence_id,
                original_sequence=sequence,
                decoy_id=decoy_id,
                decoy_sequence=decoy_seq,
                decoy_type=dtype,
                decoy_index=i
            ))

        return results

    def generate_decoys(
        self,
        sequences: List[Tuple[str, str]],
        n_decoys: Optional[int] = None,
        decoy_type: Optional[DecoyType] = None
    ) -> Tuple[List[Tuple[str, str]], List[DecoyResult]]:
        """
        Generate decoys for multiple sequences.

        Args:
            sequences: List of (sequence_id, sequence) tuples
            n_decoys: Number of decoys per sequence (uses config default if None)
            decoy_type: Type of decoy (uses config default if None)

        Returns:
            Tuple of:
                - List of output sequences (id, sequence) tuples
                - List of DecoyResult objects for tracking
        """
        n = n_decoys if n_decoys is not None else self.config.n_decoys
        dtype = decoy_type or self.config.decoy_type

        output_sequences = []
        all_results = []

        # Add original sequences if configured
        if self.config.include_original and not self.config.decoys_only:
            output_sequences.extend(sequences)

        # Generate decoys
        if n > 0:
            for seq_id, seq in sequences:
                results = self.generate_decoys_for_sequence(seq_id, seq, n, dtype)
                all_results.extend(results)

                for result in results:
                    output_sequences.append((result.decoy_id, result.decoy_sequence))

        logger.info(
            f"Generated {len(all_results)} decoys from {len(sequences)} sequences "
            f"(type: {dtype.value}, n_decoys: {n})"
        )

        if self.config.decoys_only:
            logger.info("Decoys-only mode: original sequences excluded from output")

        return output_sequences, all_results

    @staticmethod
    def is_decoy_id(sequence_id: str, prefix: str = "DECOY_") -> bool:
        """
        Check if a sequence ID indicates a decoy sequence.

        Args:
            sequence_id: Sequence ID to check
            prefix: Decoy prefix to look for

        Returns:
            True if sequence is a decoy
        """
        return sequence_id.startswith(prefix)

    @staticmethod
    def get_original_id(decoy_id: str, prefix: str = "DECOY_") -> Optional[str]:
        """
        Extract original sequence ID from decoy ID.

        Args:
            decoy_id: Decoy sequence ID
            prefix: Decoy prefix

        Returns:
            Original sequence ID or None if not a decoy
        """
        if not decoy_id.startswith(prefix):
            return None

        # Remove prefix and type tag (e.g., "DECOY_R_" or "DECOY_R0_")
        remainder = decoy_id[len(prefix):]

        # Skip type tag and optional index
        if '_' in remainder:
            parts = remainder.split('_', 1)
            if len(parts) > 1:
                return parts[1]

        return remainder

    def write_decoy_fasta(
        self,
        sequences: List[Tuple[str, str]],
        output_path: str,
        n_decoys: Optional[int] = None,
        decoy_type: Optional[DecoyType] = None
    ) -> Tuple[str, List[DecoyResult]]:
        """
        Generate decoys and write to FASTA file.

        Args:
            sequences: Input sequences
            output_path: Output FASTA file path
            n_decoys: Number of decoys per sequence
            decoy_type: Type of decoy generation

        Returns:
            Tuple of (output_path, list of DecoyResult)
        """
        output_sequences, results = self.generate_decoys(sequences, n_decoys, decoy_type)

        with open(output_path, 'w') as f:
            for seq_id, seq in output_sequences:
                f.write(f">{seq_id}\n")
                # Wrap sequence at 80 characters
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + "\n")

        logger.info(f"Wrote {len(output_sequences)} sequences to {output_path}")
        return output_path, results
