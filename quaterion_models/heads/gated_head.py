import torch
from torch.nn import Parameter

from quaterion_models.heads import EncoderHead


class GatedHead(EncoderHead):
    """
    Disables or amplifies some components of input embedding.
    This layer have minimal amount of trainable parameters and is suitable for even small training sets.
    """

    def __init__(self, input_embedding_size: int):
        super(GatedHead, self).__init__(input_embedding_size)
        self.gates = Parameter(torch.Tensor(self.input_embedding_size))
        self.reset_parameters()

    def output_size(self) -> int:
        return self.input_embedding_size

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        """

        :param input_vectors: shape: (batch_size * vector_size)
        :return: (batch_size * vector_size)
        """
        return input_vectors * torch.tanh(self.gates)

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(
            self.gates, 2.0
        )  # 2. ensures that all vector components are enabled by default
