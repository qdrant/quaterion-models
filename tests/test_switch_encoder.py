from abc import ABC
from typing import List, Any

import numpy as np
import torch
from torch import Tensor

from quaterion_models.encoder import Encoder, TensorInterchange, CollateFnType
from quaterion_models.heads.empty_head import EmptyHead
from quaterion_models.model import MetricModel
from quaterion_models.encoders.switch_encoder import SwitchEncoder


class TestEncoder(Encoder, ABC):

    def trainable(self) -> bool:
        return False

    def embedding_size(self) -> int:
        return 3

    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> 'Encoder':
        return cls()

    @classmethod
    def collate_fn(cls, batch: List[Any]) -> TensorInterchange:
        return [torch.zeros(1) for _ in batch]

    def get_collate_fn(self) -> CollateFnType:
        return self.__class__.collate_fn


class EncoderA(TestEncoder):

    def forward(self, batch: TensorInterchange) -> Tensor:
        return torch.zeros(len(batch), self.embedding_size())


class EncoderB(TestEncoder):

    def forward(self, batch: TensorInterchange) -> Tensor:
        return torch.ones(len(batch), self.embedding_size())


class MySwitchEncoder(SwitchEncoder):

    @classmethod
    def encoder_selection(cls, record: Any) -> str:
        if record == "zeros":
            return "a"
        if record == "ones":
            return "b"


def test_forward():
    encoder = MySwitchEncoder({
        'a': EncoderA(),
        'b': EncoderB()
    })

    model = MetricModel(
        encoders=encoder,
        head=EmptyHead(encoder.embedding_size())
    )
    batch = [
        "zeros",
        "zeros",
        "ones",
        "ones",
        "zeros",
        "zeros",
        "ones"
    ]

    res = model.encode(batch)

    assert res.shape[0] == len(batch)
    assert all(res[:, 0] == np.array([0., 0., 1., 1., 0., 0., 1.]))
