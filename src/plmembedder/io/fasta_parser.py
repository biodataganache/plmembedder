# io/fasta_parser.py
"""FASTA file parsing utilities."""

from typing import List, Tuple, Iterator, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FastaParser:
    """Parse FASTA files."""

    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    VALID_AA_EXTENDED = VALID_AA | set("BJOUXZ*-")

    @classmethod
    def parse(
        cls,
        fasta_path: str,
        validate: bool = True,
        remove_invalid: bool = True,
        max_sequences: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """
        Parse a FASTA file.

        Args:
            fasta_path: Path to FASTA file
            validate: Whether to validate sequences
            remove_invalid: Remove invalid characters (if False, raise error)
            max_sequences: Maximum number of sequences to load

        Returns:
            List of (sequence_id, sequence) tuples
        """
        sequences = []

        for seq_id, sequence in cls.iterate(fasta_path):
            if validate:
                sequence = cls._validate_sequence(sequence, seq_id, remove_invalid)

            if sequence:  # Skip empty sequences
                sequences.append((seq_id, sequence))

            if max_sequences and len(sequences) >= max_sequences:
                break

        logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
        return sequences

    @classmethod
    def iterate(cls, fasta_path: str) -> Iterator[Tuple[str, str]]:
        """
        Iterate over sequences in a FASTA file.

        Yields:
            (sequence_id, sequence) tuples
        """
        path = Path(fasta_path)

        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        current_id = None
        current_seq = []

        # Handle gzipped files
        if path.suffix == '.gz':
            import gzip
            opener = gzip.open
            mode = 'rt'
        else:
            opener = open
            mode = 'r'

        with opener(path, mode) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith('>'):
                    # Yield previous sequence
                    if current_id is not None:
                        yield current_id, ''.join(current_seq)

                    # Parse new header
                    current_id = line[1:].split()[0]  # Take first word as ID
                    current_seq = []
                else:
                    current_seq.append(line.upper())

            # Yield last sequence
            if current_id is not None:
                yield current_id, ''.join(current_seq)

    @classmethod
    def _validate_sequence(
        cls,
        sequence: str,
        seq_id: str,
        remove_invalid: bool
    ) -> str:
        """Validate and optionally clean a sequence."""
        invalid_chars = set(sequence) - cls.VALID_AA_EXTENDED

        if invalid_chars:
            if remove_invalid:
                logger.warning(f"Removing invalid characters {invalid_chars} from {seq_id}")
                sequence = ''.join(c for c in sequence if c in cls.VALID_AA_EXTENDED)
            else:
                raise ValueError(f"Invalid characters {invalid_chars} in sequence {seq_id}")

        # Replace extended amino acids with X
        for char in "BJOUZ":
            if char in sequence:
                sequence = sequence.replace(char, 'X')

        # Remove gaps and stops
        sequence = sequence.replace('-', '').replace('*', '')

        return sequence

    @classmethod
    def write(
        cls,
        sequences: List[Tuple[str, str]],
        output_path: str,
        line_width: int = 80
    ) -> None:
        """
        Write sequences to a FASTA file.

        Args:
            sequences: List of (id, sequence) tuples
            output_path: Output file path
            line_width: Characters per line for sequence
        """
        with open(output_path, 'w') as f:
            for seq_id, sequence in sequences:
                f.write(f">{seq_id}\n")

                # Write sequence in chunks
                for i in range(0, len(sequence), line_width):
                    f.write(sequence[i:i + line_width] + '\n')

        logger.info(f"Wrote {len(sequences)} sequences to {output_path}")
