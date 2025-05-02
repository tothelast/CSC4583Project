"""
@author: Garegin Mazmanyan
MonoT5 model for NevIR based on the original paper implementation.

This implements the best performing model from the NevIR paper (MonoT5-3B)
which achieved ~50% pairwise accuracy on the negation task.
"""
import torch
from typing import List, Tuple
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

class MonoT5Model:
    """
    A MonoT5 cross-encoder model for NevIR, based on the implementation from Pygaggle.

    MonoT5 achieves best results on the NevIR dataset by effectively handling
    the semantics of negation in both queries and documents.
    """

    def __init__(self, model_name="castorini/monot5-3b-msmarco-10k"):
        """
        Initialize the MonoT5 model.

        Args:
            model_name: The name of the MonoT5 model to use
        """
        print(f"Loading MonoT5 model: {model_name}")

        # This is just trying to use MPS - (Apple Silicon GPU) if it is available
        # since I have an M3 MacBook Pro I can take advantage of this
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "mps":
            print("Using MPS device (Apple Silicon GPU)")
        elif self.device.type == "cuda":
            print("Using CUDA device (NVIDIA GPU)")
        else:
            print("Using CPU device")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model = model.to(self.device).eval()

        self.token_false_id, self.token_true_id = self._get_prediction_token_ids(model_name)
        self.model_name = model_name

    def _get_prediction_token_ids(self, model_name):
        """Get the token IDs for the terms 'false' and 'true' for this model."""
        # This matches the exact mapping in pygaggle implementation
        prediction_tokens = {
            'castorini/monot5-base-msmarco': ['▁false', '▁true'],
            'castorini/monot5-base-msmarco-10k': ['▁false', '▁true'],
            'castorini/monot5-large-msmarco': ['▁false', '▁true'],
            'castorini/monot5-large-msmarco-10k': ['▁false', '▁true'],
            'castorini/monot5-base-med-msmarco': ['▁false', '▁true'],
            'castorini/monot5-3b-med-msmarco': ['▁false', '▁true'],
            'castorini/monot5-3b-msmarco-10k': ['▁false', '▁true']
        }

        if model_name in prediction_tokens:
            token_false, token_true = prediction_tokens[model_name]
            token_false_id = self.tokenizer.get_vocab()[token_false]
            token_true_id = self.tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id
        else:
            return 6136, 1176  # These are the T5 vocab indices for "false" and "true"

    def _greedy_decode(self, input_ids, attention_mask, length=1):
        """
        Implementation of greedy_decode from pygaggle, updated for newer transformers API
        """
        with torch.no_grad():
            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            ).to(self.device)

            encoder_outputs = self.model.get_encoder()(input_ids, attention_mask=attention_mask)

            next_token_logits = None
            for _ in range(length):
                model_inputs = {
                    "decoder_input_ids": decoder_input_ids,
                    "encoder_outputs": encoder_outputs,
                    "attention_mask": attention_mask,
                    "use_cache": True
                }

                # Forward pass
                outputs = self.model(**model_inputs)
                next_token_logits = outputs.logits[:, -1, :]

                # Greedy decoding
                next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)

            return decoder_input_ids, next_token_logits

    def _score_passages(self, query: str, passages: List[str]) -> List[float]:
        """
        Score passages using the MonoT5 model.

        Args:
            query: The query string
            passages: A list of passage strings to be scored

        Returns:
            A list of scores for each passage
        """
        with torch.no_grad():
            # Format inputs exactly as in pygaggle's T5BatchTokenizer
            inputs = [f"Query: {query} Document: {p} Relevant:" for p in passages]

            # Tokenize
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                inputs,
                max_length=512,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )

            # Move to device
            input_ids = tokenized_inputs["input_ids"].to(self.device)
            attention_mask = tokenized_inputs["attention_mask"].to(self.device)

            # Score using greedy decoding
            _, token_logits = self._greedy_decode(
                input_ids,
                attention_mask=attention_mask,
                length=1
            )

            if token_logits is not None:
                batch_scores = token_logits[:, [self.token_false_id, self.token_true_id]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)

                return batch_scores[:, 1].tolist()
            else:
                return [0.0] * len(passages)

    def rank_documents(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """
        Rank documents based on their relevance to the query.

        Args:
            query: The query string
            documents: A list of document strings to be ranked

        Returns:
            A list of tuples (doc_idx, score) sorted by score in descending order
        """
        doc_scores = self._score_passages(query, documents)
        doc_scores_with_idx = [(i, score) for i, score in enumerate(doc_scores)]
        return sorted(doc_scores_with_idx, key=lambda x: x[1], reverse=True)
