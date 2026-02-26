# embedders/base.py
"""Base class for protein language model embedders."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    sequence_id: str
    sequence: str
    embeddings: np.ndarray  # Shape: (seq_length, embedding_dim)
    attention_weights: np.ndarray = None  # Optional attention weights


class BaseEmbedder(ABC):
    """Abstract base class for protein embedders."""

    def __init__(self, model_name: str, device: str = "cuda", layer: int = -1):
        self.model_name = model_name
        self.device = device
        self.layer = layer
        self.embedding_dim = None
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Embed a single protein sequence.

        Args:
            sequence: Amino acid sequence string

        Returns:
            numpy array of shape (sequence_length, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_batch(self, sequences: List[Tuple[str, str]]) -> List[EmbeddingResult]:
        """
        Embed a batch of protein sequences.

        Args:
            sequences: List of (id, sequence) tuples

        Returns:
            List of EmbeddingResult objects
        """
        pass

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"
