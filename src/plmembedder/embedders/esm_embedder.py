# embedders/esm_embedder.py
"""ESM (Evolutionary Scale Modeling) embedder implementation."""

import torch
import numpy as np
from typing import List, Tuple
import logging

from .base import BaseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class ESMEmbedder(BaseEmbedder):
    """Embedder using Facebook's ESM models."""

    MODEL_CONFIGS = {
        "esm2_t33_650M_UR50D": {"layers": 33, "dim": 1280},
        "esm2_t30_150M_UR50D": {"layers": 30, "dim": 640},
        "esm2_t12_35M_UR50D": {"layers": 12, "dim": 480},
        "esm2_t6_8M_UR50D": {"layers": 6, "dim": 320},
        "esm1b_t33_650M_UR50S": {"layers": 33, "dim": 1280},
    }

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D",
                 device: str = "cuda", layer: int = -1):
        super().__init__(model_name, device, layer)

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")

        config = self.MODEL_CONFIGS[model_name]
        self.embedding_dim = config["dim"]
        self.num_layers = config["layers"]

        if layer == -1:
            self.layer = self.num_layers

    def load_model(self) -> None:
        """Load ESM model and batch converter."""
        try:
            import esm
        except ImportError:
            raise ImportError("Please install fair-esm: pip install fair-esm")

        logger.info(f"Loading ESM model: {self.model_name}")

        # Load model
        model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self._model = model.eval()
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()

        # Move to device
        if torch.cuda.is_available() and self.device == "cuda":
            self._model = self._model.cuda()
        else:
            self.device = "cpu"

        logger.info(f"Model loaded on {self.device}")

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Embed a single sequence."""
        results = self.embed_batch([("seq", sequence)])
        return results[0].embeddings

    def embed_batch(self, sequences: List[Tuple[str, str]],
                    max_length: int = 1024) -> List[EmbeddingResult]:
        """
        Embed a batch of sequences.

        Args:
            sequences: List of (id, sequence) tuples
            max_length: Maximum sequence length to process

        Returns:
            List of EmbeddingResult objects
        """
        if self._model is None:
            self.load_model()

        results = []

        # Truncate long sequences
        processed_sequences = []
        for seq_id, seq in sequences:
            if len(seq) > max_length:
                logger.warning(f"Truncating sequence {seq_id} from {len(seq)} to {max_length}")
                seq = seq[:max_length]
            processed_sequences.append((seq_id, seq))

        # Convert to batch
        batch_labels, batch_strs, batch_tokens = self._batch_converter(processed_sequences)

        if self.device == "cuda":
            batch_tokens = batch_tokens.cuda()

        # Get embeddings
        with torch.no_grad():
            results_dict = self._model(batch_tokens, repr_layers=[self.layer],
                                       return_contacts=False)
            embeddings = results_dict["representations"][self.layer]

        # Process each sequence
        for i, (seq_id, seq) in enumerate(processed_sequences):
            # Remove BOS and EOS tokens
            seq_len = len(seq)
            seq_embeddings = embeddings[i, 1:seq_len+1, :].cpu().numpy()

            results.append(EmbeddingResult(
                sequence_id=seq_id,
                sequence=seq,
                embeddings=seq_embeddings
            ))

        return results


class ESM2Embedder(ESMEmbedder):
    """Convenience class for ESM2 models."""

    def __init__(self, model_size: str = "650M", device: str = "cuda", layer: int = -1):
        model_map = {
            "8M": "esm2_t6_8M_UR50D",
            "35M": "esm2_t12_35M_UR50D",
            "150M": "esm2_t30_150M_UR50D",
            "650M": "esm2_t33_650M_UR50D",
        }

        if model_size not in model_map:
            raise ValueError(f"Unknown model size: {model_size}. Available: {list(model_map.keys())}")

        super().__init__(model_map[model_size], device, layer)
