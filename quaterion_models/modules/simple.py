import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationFromFnName(nn.Module):
    """Simple module constructed from function name to be used in `nn.Sequential`"""

    def __init__(self, activation_fn: str):
        """Construct a `nn.Module` that applies the specified activation function to inputs

        Args:
            activation_fn (str): Name of the activation function to apply to inputs. Must be an attribute of `torch.nn.functional`.
        """
        super().__init__()
        self._activation_fn = activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return vars(F)[self._activation_fn](inputs)
