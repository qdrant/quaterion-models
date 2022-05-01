from typing import Any, Dict

from quaterion_models.heads.stacked_projection_head import StackedProjectionHead


class WideningHead(StackedProjectionHead):
    """Implements narrow-wide-narrow architecture.

    Widen the dimensionality by a factor of `expansion_factor` and narrow it down back to
    `input_embedding_size`.

    Args:
        input_embedding_size: Dimensionality of the input to this head layer.
        expansion_factor: Widen the dimensionality by this factor in the intermediate layer.
        activation_fn: Name of the activation function to apply after the intermediate layer.
            Must be an attribute of `torch.nn.functional` and defaults to `relu`.
        dropout: Probability of Dropout. If `dropout > 0.`, apply dropout layer
            on embeddings before applying head layer transformations
    """

    def __init__(
        self,
        input_embedding_size: int,
        expansion_factor: float = 4.0,
        activation_fn: str = "relu",
        dropout: float = 0.0,
        **kwargs
    ):
        self._expansion_factor = expansion_factor
        self._activation_fn = activation_fn
        super(WideningHead, self).__init__(
            input_embedding_size=input_embedding_size,
            output_sizes=[
                int(input_embedding_size * expansion_factor),
                input_embedding_size,
            ],
            activation_fn=activation_fn,
            dropout=dropout,
        )

    def get_config_dict(self) -> Dict[str, Any]:
        config = super().get_config_dict()
        config.update(
            {
                "expansion_factor": self._expansion_factor,
            }
        )
        return config
