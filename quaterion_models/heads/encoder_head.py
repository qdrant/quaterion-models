import json
import os
from typing import Dict, Any

import torch
from torch import nn


class EncoderHead(nn.Module):
    def __init__(self, input_embedding_size: int, **kwargs):
        super(EncoderHead, self).__init__()
        self.input_embedding_size = input_embedding_size

    def output_size(self) -> int:
        raise NotImplementedError()

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Constructs savable parameters dict
        :return: Serializable parameters for __init__ of the Module
        """
        return {"input_embedding_size": self.input_embedding_size}

    def save(self, output_path):
        torch.save(self.state_dict(), os.path.join(output_path, "weights.bin"))

        with open(os.path.join(output_path, "config.json"), "w") as f_out:
            json.dump(self.get_config_dict(), f_out, indent=2)

    @classmethod
    def load(cls, input_path: str) -> "EncoderHead":
        with open(os.path.join(input_path, "config.json")) as f_in:
            config = json.load(f_in)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, "weights.bin")))
        return model
