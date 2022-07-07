from typing import Dict, Any

import torch

from torch.nn import Parameter

from quaterion_models.heads.encoder_head import EncoderHead


class SkipConnectionHead(EncoderHead):
    """Unites the idea of gated head and residual connections.

    Schema:
        .. code-block:: none

                      ├──────────┐
              ┌───────┴───────┐  │
              │  Skip-Dropout │  │
              └───────┬───────┘  │
              ┌───────┴───────┐  │
              │     Linear    │  │
              └───────┬───────┘  │
              ┌───────┴───────┐  │
              │     Gated     │  │
              └───────┬───────┘  │
                      + <────────┘
                      │

    Args:
        input_embedding_size:
            Size of the concatenated embedding, obtained from combination of all configured encoders
        dropout:
            Probability of Dropout. If `dropout > 0.`, apply dropout layer
            on embeddings before applying head layer transformations
        skip_dropout:
            Additional dropout, applied to the trainable branch only.
            Using additional dropout allows to avoid the modification of original embedding.
    """

    def __init__(
        self, input_embedding_size: int, dropout: float = 0.0, skip_dropout: float = 0.0
    ):
        super().__init__(input_embedding_size, dropout=dropout)
        self.gates = Parameter(torch.Tensor(self.input_embedding_size))
        self.reset_parameters()
        self.fc = torch.nn.Linear(input_embedding_size, input_embedding_size)
        self._skip_dropout_prob = skip_dropout
        self.skip_dropout = (
            torch.nn.Dropout(p=skip_dropout)
            if skip_dropout > 0.0
            else torch.nn.Identity()
        )

    @property
    def output_size(self) -> int:
        return self.input_embedding_size

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_vectors: shape: (batch_size, input_embedding_size)

        Returns:
            torch.Tensor: shape: (batch_size, input_embedding_size)
        """
        return (
            self.fc(self.skip_dropout(input_vectors)) * torch.sigmoid(self.gates)
            + input_vectors
        )

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(
            self.gates, -4.0
        )  # -4. ensures that all vector components are disabled by default

    def get_config_dict(self) -> Dict[str, Any]:
        config = super().get_config_dict()
        config.update({"skip_dropout": self._skip_dropout_prob})
        return config
