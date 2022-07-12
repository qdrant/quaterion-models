import os
import tempfile
from typing import List, Any

import torch
import torch.nn as nn
from torch import Tensor


from quaterion_models.encoders import Encoder
from quaterion_models.model import SimilarityModel
from quaterion_models.heads.sequential_head import SequentialHead
from quaterion_models.types import TensorInterchange, CollateFnType


BATCH_SIZE = 3
INPUT_EMBEDDING_SIZE = 5
HIDDEN_EMBEDDING_SIZE = 6
OUTPUT_EMBEDDING_SIZE = 7


def test_forward():
    head = SequentialHead(
        nn.Linear(INPUT_EMBEDDING_SIZE, HIDDEN_EMBEDDING_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_EMBEDDING_SIZE, OUTPUT_EMBEDDING_SIZE),
        output_size=OUTPUT_EMBEDDING_SIZE,
    )

    batch = torch.rand(BATCH_SIZE, INPUT_EMBEDDING_SIZE)
    res = head.forward(batch)

    assert res.shape == (BATCH_SIZE, OUTPUT_EMBEDDING_SIZE)


class CustomEncoder(Encoder):
    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return cls()

    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return INPUT_EMBEDDING_SIZE

    @classmethod
    def collate_fn(cls, batch: List[Any]):
        return torch.stack(batch)

    def get_collate_fn(self) -> CollateFnType:
        return self.__class__.collate_fn

    def forward(self, batch: TensorInterchange) -> Tensor:
        return batch


def test_save_and_load():
    encoder = CustomEncoder()
    head = SequentialHead(
        nn.Linear(INPUT_EMBEDDING_SIZE, HIDDEN_EMBEDDING_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_EMBEDDING_SIZE, OUTPUT_EMBEDDING_SIZE),
        output_size=OUTPUT_EMBEDDING_SIZE,
    )

    model = SimilarityModel(encoders=encoder, head=head)
    tempdir = tempfile.TemporaryDirectory()

    model.save(tempdir.name)

    config_path = os.path.join(tempdir.name, "config.json")

    assert os.path.exists(config_path)

    batch = torch.rand(BATCH_SIZE, INPUT_EMBEDDING_SIZE)
    origin_output = model.encode(batch, to_numpy=False)

    loaded_model = SimilarityModel.load(tempdir.name)

    assert model.encoders.keys() == loaded_model.encoders.keys()
    assert [type(encoder) for encoder in model.encoders.values()] == [
        type(encoder) for encoder in loaded_model.encoders.values()
    ]

    assert type(model.head) == type(loaded_model.head)
    assert torch.allclose(origin_output, loaded_model.encode(batch, to_numpy=False))


if __name__ == "__main__":
    test_save_and_load()
