from multiprocessing import Pool

import torch

from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.model import MetricModel


class LambdaHead(EncoderHead):

    def __init__(self):
        super(LambdaHead, self).__init__()
        self.my_lambda = lambda x: "hello"

    def output_size(self) -> int:
        return 0

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return input_vectors


class Tst:
    def __init__(self, foo):
        self.foo = foo

    def bar(self, x):
        return self.foo(x)


def test_get_collate_fn():

    model = MetricModel(encoders={}, head=LambdaHead())

    tester = Tst(foo=model.get_collate_fn())

    with Pool(2) as pool:
        res = pool.map(tester.bar, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    assert res == [{}] * 10
