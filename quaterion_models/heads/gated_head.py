import torch
from torch.nn import Parameter

from quaterion_models.heads import EncoderHead


class GatedHead(EncoderHead):
    """Disables or amplifies some components of input embedding.

    This layer has minimal amount of trainable parameters and is suitable for even small training
    sets.
    """

    def __init__(self, input_embedding_size: int, dropout: float = 0.0):
        super(GatedHead, self).__init__(input_embedding_size, dropout=dropout)
        self.gates = Parameter(torch.Tensor(self.input_embedding_size))
        self.reset_parameters()

    @property
    def output_size(self) -> int:
        return self.input_embedding_size

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input_vectors: shape: (batch_size, vector_size)

        Returns:
            Tensor: (batch_size, vector_size)
        """
        return input_vectors * torch.tanh(self.gates)

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(
            self.gates, 2.0
        )  # 2. ensures that all vector components are enabled by default
