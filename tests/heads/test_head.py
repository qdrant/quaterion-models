import os
import tempfile
from typing import List, Any

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from quaterion_models import SimilarityModel
from quaterion_models.encoders import Encoder
from quaterion_models.heads import (
    SequentialHead,
    GatedHead,
    WideningHead,
    SkipConnectionHead,
    EmptyHead,
    SoftmaxEmbeddingsHead,
)
from quaterion_models.types import CollateFnType, TensorInterchange


BATCH_SIZE = 3
INPUT_EMBEDDING_SIZE = 5
HIDDEN_EMBEDDING_SIZE = 7
OUTPUT_EMBEDDING_SIZE = 10

_HEADS = (
    SequentialHead(
        nn.Linear(INPUT_EMBEDDING_SIZE, HIDDEN_EMBEDDING_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_EMBEDDING_SIZE, OUTPUT_EMBEDDING_SIZE),
        output_size=OUTPUT_EMBEDDING_SIZE,
    ),
    GatedHead(INPUT_EMBEDDING_SIZE),
    WideningHead(INPUT_EMBEDDING_SIZE),
    SkipConnectionHead(INPUT_EMBEDDING_SIZE),
    EmptyHead(INPUT_EMBEDDING_SIZE),
    SoftmaxEmbeddingsHead(
        output_groups=2,
        output_size_per_group=OUTPUT_EMBEDDING_SIZE,
        input_embedding_size=INPUT_EMBEDDING_SIZE,
    ),
)
HEADS = {head_.__class__.__name__: head_ for head_ in _HEADS}


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


@pytest.mark.parametrize("head", HEADS.values(), ids=HEADS.keys())
def test_save_and_load(head):
    encoder = CustomEncoder()

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


@pytest.mark.parametrize("head", HEADS.values(), ids=HEADS.keys())
def test_forward_shape(head):
    batch = torch.rand(BATCH_SIZE, INPUT_EMBEDDING_SIZE)
    res = head.forward(batch)
    assert res.shape == (BATCH_SIZE, head.output_size)
