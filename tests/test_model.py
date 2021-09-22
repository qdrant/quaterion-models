import json
import os
import tempfile
from multiprocessing import Pool
from typing import List, Any

import torch
from torch import Tensor

from quaterion_models.encoder import Encoder, TensorInterchange, CollateFnType
from quaterion_models.heads.empty_head import EmptyHead
from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.model import MetricModel

TEST_EMB_SIZE = 5


class LambdaHead(EncoderHead):

    def __init__(self):
        super(LambdaHead, self).__init__(TEST_EMB_SIZE)
        self.my_lambda = lambda x: "hello"

    def output_size(self) -> int:
        return 0

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return input_vectors


class TestEncoder(Encoder):

    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> 'Encoder':
        return cls()

    def __init__(self):
        super().__init__()
        self.unpickable = lambda x: x + 1

    def trainable(self) -> bool:
        return False

    def embedding_size(self) -> int:
        return TEST_EMB_SIZE

    @classmethod
    def collate_fn(cls, batch: List[Any]):
        return torch.rand(len(batch), TEST_EMB_SIZE)

    def get_collate_fn(self) -> CollateFnType:
        return self.__class__.collate_fn

    def forward(self, batch: TensorInterchange) -> Tensor:
        return batch


class Tst:
    def __init__(self, foo):
        self.foo = foo

    def bar(self, x):
        return self.foo(x)


def test_get_collate_fn():
    model = MetricModel(encoders={
        "test": TestEncoder()
    }, head=LambdaHead())

    tester = Tst(foo=model.get_collate_fn())

    with Pool(2) as pool:
        res = pool.map(tester.bar, [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10]
        ])

    assert len(res) == 4

    first_batch = res[0]

    assert 'test' in first_batch

    tensor = first_batch['test']

    assert tensor.shape == (3, TEST_EMB_SIZE)


def test_model_save_and_load():
    tempdir = tempfile.TemporaryDirectory()
    model = MetricModel(
        encoders={
            "test": TestEncoder()
        },
        head=EmptyHead(100)
    )

    model.save(tempdir.name)

    config_path = os.path.join(tempdir.name, 'config.json')

    assert os.path.exists(config_path)

    loaded_model = MetricModel.load(tempdir.name)

    assert model.encoders.keys() == loaded_model.encoders.keys()
    assert [type(encoder) for encoder in model.encoders.values()] == \
           [type(encoder) for encoder in loaded_model.encoders.values()]

    assert type(model.head) == type(loaded_model.head)
