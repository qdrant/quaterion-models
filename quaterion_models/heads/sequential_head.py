import json
import os
from typing import Any, Dict, Iterator, Union

import torch
import torch.nn as nn

from quaterion_models.heads.encoder_head import EncoderHead


class SequentialHead(EncoderHead):
    """A `torch.nn.Sequential`-like head layer that you can freely add any layers.

    Unlike `torch.nn.Sequential`, it also expects the output size to be passed
    as a required  keyword-only argument. It is required because some loss functions
    may need this information.

    Args:
        args: Any sequence of `torch.nn.Module` instances. See `torch.nn.Sequential` for more info.
        output_size: Final output dimension from this head.

    Examples::

        head = SequentialHead(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            output_size=30
        )
    """

    def __init__(self, *args, output_size: int):
        super().__init__(None)
        self._sequential = nn.Sequential(*args)
        self._output_size = output_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, input_vectors: torch.Tensor, meta=None) -> torch.Tensor:
        """Forward pass for this head layer.

        Just like `torch.nn.Sequential`, it passes the input to the first module,
        and the output of each module is input to the next. The final output of this head layer is
        the output from the last module in the sequence.

        Args:
            input_vectors: Batch of input vectors.
            meta: Optional metadata for this batch.

        Returns:
            Output from the last module in the sequence.
        """
        return self._sequential.forward(input_vectors)

    def append(self, module: nn.Module) -> "SequentialHead":
        self._sequential.append(module)
        return self

    def get_config_dict(self) -> Dict[str, Any]:
        """Constructs savable params dict

        Returns:
            Serializable parameters for __init__ of the Module
        """
        return {
            "output_size": self._output_size,
        }

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return input_vectors

    def save(self, output_path):
        torch.save(self._sequential, os.path.join(output_path, "weights.bin"))

        with open(os.path.join(output_path, "config.json"), "w") as f_out:
            json.dump(self.get_config_dict(), f_out, indent=2)

    @classmethod
    def load(cls, input_path: str) -> "EncoderHead":
        with open(os.path.join(input_path, "config.json")) as f_in:
            config = json.load(f_in)
        sequential = torch.load(
            os.path.join(input_path, "weights.bin"), map_location="cpu"
        )
        model = cls(*sequential, **config)
        return model

    def __getitem__(self, idx) -> Union[nn.Sequential, nn.Module]:
        return self._sequential[idx]

    def __delitem__(self, idx: Union[slice, int]) -> None:
        return self._sequential.__delitem(idx)

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        return self._sequential.__setitem__(idx, module)

    def __len__(self) -> int:
        return self._sequential.__len__()

    def __dir__(self):
        return self._sequential.__dir__()

    def __iter__(self) -> Iterator[nn.Module]:
        return self._sequential.__iter__()
