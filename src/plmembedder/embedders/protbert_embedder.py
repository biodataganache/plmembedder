# embedders/protbert_embedder.py
"""ProtBert embedder implementation."""

import torch
import numpy as np
from typing import List, Tuple
import logging

from .base import BaseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class ProtBertEmbedder(BaseEmbedder):
    """Embedder using ProtBert models from Hugging Face."""

    MODEL_CONFIGS = {
        "Rostlab/prot_bert": {"dim": 1024},
        "Rostlab/prot_bert_bfd": {"dim": 1024},
    }

    def __init__(self, model_name: str = "Rostlab/prot_bert",
                 device: str = "cuda", layer: int = -1):
        super().__init__(model_name, device, layer)

        if model_name not in self.MODEL_CONFIGS:
            logger.warning(f"Model {model_name} not in known configs, assuming dim=1024")
            self.embedding_dim = 1024
        else:
            self.embedding_dim = self.MODEL_CONFIGS[model_name]["dim"]

    def load_model(self) -> None:
        """Load ProtBert model and tokenizer."""
        try:
            from transformers import BertModel, BertTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        logger.info(f"Loading ProtBert model: {self.model_name}")

        self._tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self._model = BertModel.from_pretrained(self.model_name)
        self._model.eval()

        if torch.cuda.is_available() and self.device == "cuda":
            self._model = self._model.cuda()
        else:
            self.device = "cpu"

        logger.info(f"Model loaded on {self.device}")

    def _preprocess_sequence(self, sequence: str) -> str:
        """Add spaces between amino acids for ProtBert."""
        return " ".join(list(sequence))

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Embed a single sequence."""
        results = self.embed_batch([("seq", sequence)])
        return results[0].embeddings

    def embed_batch(self, sequences: List[Tuple[str, str]],
                    max_length: int = 1024) -> List[EmbeddingResult]:
        """Embed a batch of sequences."""
        if self._model is None:
            self.load_model()

        results = []

        # Preprocess sequences
        processed_sequences = []
        for seq_id, seq in sequences:
            if len(seq) > max_length:
                logger.warning(f"Truncating sequence {seq_id} from {len(seq)} to {max_length}")
                seq = seq[:max_length]
            processed_sequences.append((seq_id, seq, self._preprocess_sequence(seq)))

        # Tokenize
        tokenized = self._tokenizer(
            [s[2] for s in processed_sequences],
            padding=True,
            truncation=True,
            max_length=max_length + 2,  # Account for special tokens
            return_tensors="pt"
        )

        if self.device == "cuda":
            tokenized = {k: v.cuda() for k, v in tokenized.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**tokenized, output_hidden_states=True)

            if self.layer == -1:
                embeddings = outputs.last_hidden_state
            else:
                embeddings = outputs.hidden_states[self.layer]

        # Process each sequence
        for i, (seq_id, seq, _) in enumerate(processed_sequences):
            seq_len = len(seq)
            # Remove CLS and SEP tokens
            seq_embeddings = embeddings[i, 1:seq_len+1, :].cpu().numpy()

            results.append(EmbeddingResult(
                sequence_id=seq_id,
                sequence=seq,
                embeddings=seq_embeddings
            ))

        return results
