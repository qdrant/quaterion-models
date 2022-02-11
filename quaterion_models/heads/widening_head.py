from typing import Any, Dict

from quaterion_models.heads.stacked_projection_head import StackedProjectionHead


class WideningHead(StackedProjectionHead):
    """Implements narrow-wide-narrow architecture."""

    def __init__(
        self,
        input_embedding_size: int,
        expansion_factor: float = 4.0,
        activation_fn: str = "relu",
    ):
        """Widen the dimensionality by a factor of `expension_factor` and narrow it down back to `input_embedding_size`.

        Args:
            input_embedding_size (int): Dimensionality of the input to this head layer.
            expansion_factor (float, optional): Widen the dimensionality by this factor in the intermediate layer. Defaults to 4.0.
            activation_fn (str, optional): Name of the activation function to apply after the intermediate layer. Must be an attribute of `torch.nn.functional` and defaults to `relu`.
        """
        self._expansion_factor = expansion_factor
        self._activation_fn = activation_fn
        super(WideningHead, self).__init__(
            input_embedding_size,
            [int(input_embedding_size * expansion_factor), input_embedding_size],
            activation_fn,
        )

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "input_embedding_size": self.input_embedding_size,
            "expansion_factor": self._expansion_factor,
            "activation_fn": self._activation_fn,
        }
