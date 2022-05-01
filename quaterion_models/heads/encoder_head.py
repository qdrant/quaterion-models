import json
import os
from typing import Dict, Any

import torch
from torch import nn


class EncoderHead(nn.Module):
    """Base class for the final layer of fine-tuned model.
    EncoderHead is the only trainable component in case of frozen encoders.

    Args:
        input_embedding_size:
            Size of the concatenated embedding, obtained from combination of all configured encoders
        dropout:
            Probability of Dropout. If `dropout > 0.`, apply dropout layer
            on embeddings before applying head layer transformations
        **kwargs:
    """

    def __init__(self, input_embedding_size: int, dropout: float = 0.0, **kwargs):
        super(EncoderHead, self).__init__()
        self.input_embedding_size = input_embedding_size
        self._dropout_prob = dropout
        self.dropout = (
            torch.nn.Dropout(p=dropout) if dropout > 0.0 else torch.nn.Identity()
        )

    @property
    def output_size(self) -> int:
        raise NotImplementedError()

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        """Apply head-specific transformations to the embeddings tensor.
        Called as part of `forward` function, but with generic wrappings

        Args:
            input_vectors: Concatenated embeddings of all encoders. Shape: (batch_size, self.input_embedding_size)

        Returns:
            Final embeddings for a batch: (batch_size, self.output_size)
        """
        raise NotImplementedError()

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return self.transform(self.dropout(input_vectors))

    def get_config_dict(self) -> Dict[str, Any]:
        """Constructs savable params dict

        Returns:
            Serializable parameters for __init__ of the Module
        """
        return {
            "input_embedding_size": self.input_embedding_size,
            "dropout": self._dropout_prob,
        }

    def save(self, output_path):
        torch.save(self.state_dict(), os.path.join(output_path, "weights.bin"))

        with open(os.path.join(output_path, "config.json"), "w") as f_out:
            json.dump(self.get_config_dict(), f_out, indent=2)

    @classmethod
    def load(cls, input_path: str) -> "EncoderHead":
        with open(os.path.join(input_path, "config.json")) as f_in:
            config = json.load(f_in)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, "weights.bin")))
        return model
