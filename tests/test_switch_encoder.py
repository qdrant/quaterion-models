from abc import ABC
from typing import Any, List

import numpy as np
import torch
from torch import Tensor

from quaterion_models.encoders import Encoder
from quaterion_models.encoders.switch_encoder import SwitchEncoder
from quaterion_models.heads.empty_head import EmptyHead
from quaterion_models.model import SimilarityModel
from quaterion_models.types import CollateFnType, TensorInterchange


class CustomEncoder(Encoder, ABC):
    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return 3

    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return cls()

    @classmethod
    def collate_fn(cls, batch: List[Any]) -> TensorInterchange:
        return [torch.zeros(1) for _ in batch]

    def get_collate_fn(self) -> CollateFnType:
        return self.__class__.collate_fn


class EncoderA(CustomEncoder):
    def forward(self, batch: TensorInterchange) -> Tensor:
        return torch.zeros(len(batch), self.embedding_size)


class EncoderB(CustomEncoder):
    def forward(self, batch: TensorInterchange) -> Tensor:
        return torch.ones(len(batch), self.embedding_size)


class CustomSwitchEncoder(SwitchEncoder):
    @classmethod
    def encoder_selection(cls, record: Any) -> str:
        if record == "zeros":
            return "a"
        if record == "ones":
            return "b"


def test_forward():
    encoder = CustomSwitchEncoder({"a": EncoderA(), "b": EncoderB()})

    model = SimilarityModel(encoders=encoder, head=EmptyHead(encoder.embedding_size))
    batch = ["zeros", "zeros", "ones", "ones", "zeros", "zeros", "ones"]

    res = model.encode(batch)

    assert res.shape[0] == len(batch)
    assert all(res[:, 0] == np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]))
