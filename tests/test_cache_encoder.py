from typing import Any, List

import numpy as np
import torch
from torch import Tensor

from quaterion_models.encoder import Encoder, TensorInterchange, CollateFnType
from quaterion_models.encoders.cache_encoder import CacheEncoder, EmbeddingCacheAccessor


class TestEncoder(Encoder):

    def forward(self, batch: TensorInterchange) -> Tensor:
        return torch.stack([
            torch.zeros(self.embedding_size()) + val for val in batch
        ])

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
        return [record['val'] for record in batch]

    def get_collate_fn(self) -> CollateFnType:
        return self.__class__.collate_fn


class SimpleCache(EmbeddingCacheAccessor):
    cache = {}

    def get_embeddings(self, keys: List[str]) -> np.ndarray:
        return np.stack([self.cache[key] for key in keys])

    def put_embeddings(self, keys: List[str], embeddings: np.ndarray):
        for key, embedding in zip(keys, embeddings):
            self.cache[key] = embedding

    def check_keys(self, keys: List[str]) -> List[bool]:
        return [(key in self.cache) for key in keys]


class TestCacheEncoder(CacheEncoder):

    @classmethod
    def get_cache_obj(cls) -> EmbeddingCacheAccessor:
        return SimpleCache()

    @classmethod
    def key_extraction_fn(cls, record: Any) -> str:
        return record['name']


def test_cached_forward():
    encoder = TestEncoder()
    cache_encoder = TestCacheEncoder(encoder=encoder)

    records_1 = [
        {"name": 'aaa1', "val": 1},
        {"name": 'bbb1', "val": 2},
        {"name": 'ccc1', "val": 3},
        {"name": 'ddd1', "val": 4},
        {"name": 'eee1', "val": 5},
        {"name": 'fff1', "val": 6},
    ]

    records_2 = [
        {"name": 'aaa1', "val": 1},
        {"name": 'bbb2', "val": 12},
        {"name": 'ccc2', "val": 13},
        {"name": 'ddd1', "val": 4},
        {"name": 'eee2', "val": 15},
        {"name": 'fff1', "val": 6},
    ]

    collater = cache_encoder.get_collate_fn()
    batch1 = collater(records_1)

    assert len(batch1['cached_keys']) == 0

    embeddings1 = cache_encoder.forward(batch1)

    assert embeddings1.shape[0] == len(records_1)
    assert embeddings1.shape[1] == encoder.embedding_size()

    collater = cache_encoder.get_collate_fn()
    batch2 = collater(records_2)

    assert len(batch2['cached_keys']) > 0

    embeddings2 = cache_encoder.forward(batch2)

    assert embeddings2.shape[0] == len(records_2)
    assert embeddings2.shape[1] == encoder.embedding_size()

    for record, embd in zip(records_2, embeddings2[:, 0]):
        assert record['val'] == int(embd)
