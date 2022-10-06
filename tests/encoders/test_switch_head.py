import tempfile

import torch

from quaterion_models.heads import EncoderHead
from quaterion_models.heads.switch_head import SwitchHead


class FakeHeadA(EncoderHead):
    @property
    def output_size(self) -> int:
        return 3

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return input_vectors + torch.tensor(
            [[0, 0, 1] for _ in range(input_vectors.shape[0])]
        )


class FakeHeadB(EncoderHead):
    @property
    def output_size(self) -> int:
        return 3

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return input_vectors + torch.tensor(
            [[0, 2, 0] for _ in range(input_vectors.shape[0])]
        )


def test_save_and_load():
    head = SwitchHead(
        options={"a": FakeHeadA(3), "b": FakeHeadB(3)},
        input_embedding_size=3,
    )

    temp_dir = tempfile.TemporaryDirectory()
    head.save(temp_dir.name)

    loaded_head = SwitchHead.load(temp_dir.name)

    print(loaded_head)


def test_forward():
    head = SwitchHead(
        options={"a": FakeHeadA(3), "b": FakeHeadB(3)},
        input_embedding_size=3,
    )

    batch = torch.tensor(
        [
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
            [5, 0, 0],
        ]
    )

    meta = [
        {"encoder": "b"},
        {"encoder": "a"},
        {"encoder": "b"},
        {"encoder": "b"},
        {"encoder": "a"},
    ]

    res = head.forward(batch, meta)

    assert res.shape[0] == batch.shape[0]
    assert res.shape[1] == 3

    assert res[0][0] == 1
    assert res[1][0] == 2
    assert res[2][0] == 3
    assert res[3][0] == 4
    assert res[4][0] == 5

    assert res[0][1] == 2
    assert res[1][1] == 0
    assert res[2][1] == 2
    assert res[3][1] == 2
    assert res[4][1] == 0

    assert res[0][2] == 0
    assert res[1][2] == 1
    assert res[2][2] == 0
    assert res[3][2] == 0
    assert res[4][2] == 1


test_save_and_load()
