import json
import os
from typing import Any, Dict, List

import torch
from torch import Tensor

from quaterion_models.encoders.switch_encoder import inverse_permutation
from quaterion_models.heads import EncoderHead
from quaterion_models.utils import restore_class, save_class_import


class SwitchHead(EncoderHead):
    """Encoder which switches between different heads based on the metadata
    Useful in combination with the SwitchEncoder for training multimodal models

    Args:
        options: dict of heads. Choice of head is based on the metadata key
    """

    def __init__(
        self, options: Dict[str, EncoderHead], input_embedding_size: int, **kwargs
    ):
        super().__init__(input_embedding_size, dropout=0.0, **kwargs)
        self._heads = options
        for key, head in self._heads.items():
            self.add_module(key, head)

    @property
    def output_size(self) -> int:
        return next(iter(self._heads.values())).output_size

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        pass

    def forward(
        self, input_vectors: torch.Tensor, meta: List[Any] = None
    ) -> torch.Tensor:
        # Shape: [batch_size x input_embedding_size]
        dropout_input = self.dropout(input_vectors)

        switch_mask = dict((key, []) for key in self._heads.keys())
        switch_ordering = dict((key, []) for key in self._heads.keys())
        for i, m in enumerate(meta):
            switch_ordering[m["encoder"]].append(i)
            for key, mask in switch_mask.items():
                mask.append(int(key == m["encoder"]))

        head_outputs = []
        ordering = []
        for key, mask in switch_mask.items():
            # Shape: [batch_size]
            mask = torch.tensor(mask, dtype=torch.bool, device=input_vectors.device)
            head_outputs.append(self._heads[key].transform(dropout_input[mask]))
            ordering += switch_ordering[key]

        ordering_tensor: Tensor = inverse_permutation(torch.tensor(ordering))
        # Shape: [batch_size x output_size]
        return torch.cat(head_outputs)[ordering_tensor]

    def get_config_dict(self) -> Dict[str, Any]:
        """Constructs savable params dict

        Returns:
            Serializable parameters for __init__ of the Module
        """
        return {
            "heads": {
                k: {"config": v.get_config_dict(), "class": save_class_import(v)}
                for k, v in self._heads.items()
            },
            "input_embedding_size": self.input_embedding_size,
        }

    @classmethod
    def load(cls, input_path: str) -> "EncoderHead":
        with open(os.path.join(input_path, "config.json")) as f_in:
            config = json.load(f_in)

        heads_config = config.pop("heads")

        heads = dict(
            (key, restore_class(head_config["class"])(**head_config["config"]))
            for key, head_config in heads_config.items()
        )

        model = cls(options=heads, **config)
        model.load_state_dict(torch.load(os.path.join(input_path, "weights.bin")))
        return model
