from typing import Optional

import torch

from torch import nn

from quaterion_models.heads import GatedHead


class SkipConnectionHead(GatedHead):
    """Unites the idea of gated head and residual connections

    Args:
        input_embedding_size: size of input embedding
        output_size: size of output
        downsample: nn.Module instance to change identity size to successfully
            perform addition with output at the end of the block

    """

    def __init__(
        self,
        input_embedding_size: int,
        output_size: Optional[int] = None,
        downsample: Optional[nn.Module] = None,
    ):
        if not output_size:
            output_size = input_embedding_size
        if output_size != input_embedding_size and not downsample:
            raise ValueError(
                "`downsample` has to be specified if `output_size` "
                "is set and is not equal with `input_embedding_size`"
            )

        super().__init__(output_size)
        self._input_embedding_size = input_embedding_size
        self._output_embedding_size = output_size
        self.fc = torch.nn.Linear(input_embedding_size, self._output_size)
        self.downsample = downsample

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_vectors: shape: (batch_size, input_embedding_size)

        Returns:
            torch.Tensor: shape: (batch_size, output_size)
        """
        identity = input_vectors
        if self.downsample:
            identity = self.downsample(identity)
        return self.fc(input_vectors) * torch.tanh(self.gates) + identity

    def get_config_dict(self):
        config = super().get_config_dict()
        config.update({"output_size": self.output_size, "downsample": self.downsample})
        return config
