# embedders/prott5_embedder.py
"""ProtT5 embedder implementation."""

import torch
import numpy as np
from typing import List, Tuple
import logging
import re

from .base import BaseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class ProtT5Embedder(BaseEmbedder):
    """Embedder using ProtT5 models from Hugging Face."""

    MODEL_CONFIGS = {
        "Rostlab/prot_t5_xl_half_uniref50-enc": {"dim": 1024},
        "Rostlab/prot_t5_xl_uniref50": {"dim": 1024},
        "Rostlab/prot_t5_xxl_uniref50": {"dim": 1024},
        "Rostlab/prot_t5_base_mt_uniref50": {"dim": 768},
    }

    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
        device: str = "cuda",
        layer: int = -1
    ):
        super().__init__(model_name, device, layer)

        if model_name not in self.MODEL_CONFIGS:
            logger.warning(f"Model {model_name} not in known configs, assuming dim=1024")
            self.embedding_dim = 1024
        else:
            self.embedding_dim = self.MODEL_CONFIGS[model_name]["dim"]

    def load_model(self) -> None:
        """Load ProtT5 model and tokenizer."""
        try:
            from transformers import T5Tokenizer, T5EncoderModel
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers sentencepiece"
            )

        logger.info(f"Loading ProtT5 model: {self.model_name}")

        self._tokenizer = T5Tokenizer.from_pretrained(
            self.model_name,
            do_lower_case=False
        )
        self._model = T5EncoderModel.from_pretrained(self.model_name)
        self._model.eval()

        # Handle half-precision model
        if "half" in self.model_name:
            self._model = self._model.half()

        if torch.cuda.is_available() and self.device == "cuda":
            self._model = self._model.cuda()
        else:
            self.device = "cpu"

        logger.info(f"Model loaded on {self.device}")

    def _preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess sequence for ProtT5.

        ProtT5 expects sequences with spaces between amino acids
        and rare amino acids replaced.
        """
        # Replace rare amino acids
        sequence = sequence.upper()
        sequence = re.sub(r"[UZOB]", "X", sequence)

        # Add spaces between amino acids
        return " ".join(list(sequence))

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Embed a single sequence."""
        results = self.embed_batch([("seq", sequence)])
        return results[0].embeddings

    def embed_batch(
        self,
        sequences: List[Tuple[str, str]],
        max_length: int = 1024
    ) -> List[EmbeddingResult]:
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

        # Process sequences
        processed_sequences = []
        for seq_id, seq in sequences:
            if len(seq) > max_length:
                logger.warning(
                    f"Sequence {seq_id} truncated from {len(seq)} to {max_length}"
                )
                seq = seq[:max_length]

            processed_seq = self._preprocess_sequence(seq)
            processed_sequences.append((seq_id, seq, processed_seq))

        # Tokenize
        tokenized = self._tokenizer.batch_encode_plus(
            [s[2] for s in processed_sequences],
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        )

        if self.device == "cuda":
            tokenized = {k: v.cuda() for k, v in tokenized.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self._model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"]
            )
            embeddings = outputs.last_hidden_state

        # Extract per-sequence embeddings
        for i, (seq_id, seq, _) in enumerate(processed_sequences):
            seq_len = len(seq)

            # Get attention mask to find actual sequence length
            attention_mask = tokenized["attention_mask"][i]
            actual_len = attention_mask.sum().item()

            # Remove padding (ProtT5 doesn't have CLS/SEP like BERT)
            # The output length matches input length including spaces
            seq_embeddings = embeddings[i, :seq_len, :].cpu().numpy()

            # Convert to float32 if using half-precision model
            if seq_embeddings.dtype == np.float16:
                seq_embeddings = seq_embeddings.astype(np.float32)

            results.append(EmbeddingResult(
                sequence_id=seq_id,
                sequence=seq,
                embeddings=seq_embeddings
            ))

        return results


class ProtT5XLEmbedder(ProtT5Embedder):
    """Convenience class for ProtT5-XL model (recommended for most uses)."""

    def __init__(self, device: str = "cuda", layer: int = -1, half_precision: bool = True):
        if half_precision:
            model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
        else:
            model_name = "Rostlab/prot_t5_xl_uniref50"

        super().__init__(model_name, device, layer)
