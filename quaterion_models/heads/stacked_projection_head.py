from typing import Any, Dict, List

import torch
import torch.nn as nn

from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.modules import ActivationFromFnName


class StackedProjectionHead(EncoderHead):
    """Stacks any number of projection layers with specified output sizes.

    Args:
        input_embedding_size: Dimensionality of the input to this stack of layers.
        output_sizes: List of output sizes for each one of the layers stacked.
        activation_fn: Name of the activation function to apply between the layers stacked.
            Must be an attribute of `torch.nn.functional` and defaults to `relu`.
        dropout: Probability of Dropout. If `dropout > 0.`, apply dropout layer
            on embeddings before applying head layer transformations
    """

    def __init__(
        self,
        input_embedding_size: int,
        output_sizes: List[int],
        activation_fn: str = "relu",
        dropout: float = 0.0,
    ):
        super(StackedProjectionHead, self).__init__(
            input_embedding_size, dropout=dropout
        )
        self._output_sizes = output_sizes
        self._activation_fn = activation_fn

        modules = [nn.Linear(input_embedding_size, self._output_sizes[0])]

        if len(self._output_sizes) > 1:
            for i in range(1, len(self._output_sizes)):
                modules.extend(
                    [
                        ActivationFromFnName(self._activation_fn),
                        nn.Linear(self._output_sizes[i - 1], self._output_sizes[i]),
                    ]
                )

        self._stack = nn.Sequential(*modules)

    @property
    def output_size(self) -> int:
        return self._output_sizes[-1]

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return self._stack(input_vectors)

    def get_config_dict(self) -> Dict[str, Any]:
        config = super().get_config_dict()
        config.update(
            {"output_sizes": self._output_sizes, "activation_fn": self._activation_fn}
        )

        return config
