import torch

from torch.nn import Parameter

from quaterion_models.heads import EncoderHead


class SkipConnectionHead(EncoderHead):
    """Unites the idea of gated head and residual connections."""

    def __init__(self, input_embedding_size: int):
        super().__init__(input_embedding_size)
        self.gates = Parameter(torch.Tensor(self.input_embedding_size))
        self.reset_parameters()

        self.fc = torch.nn.Linear(input_embedding_size, input_embedding_size)

    @property
    def output_size(self) -> int:
        return self.input_embedding_size

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_vectors: shape: (batch_size, input_embedding_size)

        Returns:
            torch.Tensor: shape: (batch_size, input_embedding_size)
        """
        return self.fc(input_vectors) * torch.sigmoid(self.gates) + input_vectors

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(
            self.gates, -4.0
        )  # -4. ensures that all vector components are disabled by default
