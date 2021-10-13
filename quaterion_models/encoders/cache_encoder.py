import json
import os
from typing import Any, List, Callable

import numpy as np
import torch
from torch import Tensor

from quaterion_models.encoder import Encoder, TensorInterchange, CollateFnType
from quaterion_models.utils.classes import save_class_import, restore_class
from quaterion_models.utils.tensors import inverse_permutation


class EmbeddingCacheAccessor:

    def get_embeddings(self, keys: List[str]) -> np.ndarray:
        raise NotImplementedError()

    def put_embeddings(self, keys: List[str], embeddings: np.ndarray):
        raise NotImplementedError()

    def check_keys(self, keys: List[str]) -> List[bool]:
        raise NotImplementedError()


class CachedCollater:
    def __init__(self,
                 key_fn: Callable[[Any], str],
                 collate_fn: CollateFnType,
                 cache_getter: Callable[[], EmbeddingCacheAccessor]):
        self.key_fn = key_fn
        self.collate_fn = collate_fn
        self.cache_getter = cache_getter
        self.cache = None

    def __call__(self, batch: List[Any]) -> TensorInterchange:
        if self.cache is None:
            self.cache = self.cache_getter()

        keys = []
        for record in batch:
            keys.append(self.key_fn(record))

        res = {
            'cached_keys': [],  # Keys of cached embeddings
            'cached_order': [],  # Order of `cached_keys` in batch
            'records': None,  # New records, processed with regular collate
            'records_order': [],  # Order of regular records in batch
            'records_keys': []  # Cache keys of regular records to be cached
        }
        cache_presence = self.cache.check_keys(keys)
        new_records = []
        for idx, (key, is_cached, record) in enumerate(zip(keys, cache_presence, batch)):
            if is_cached:
                res['cached_keys'].append(key)
                res['cached_order'].append(idx)
            else:
                res['records_order'].append(idx)
                res['records_keys'].append(key)
                new_records.append(record)

        res['records'] = self.collate_fn(new_records)
        return res


class CacheEncoder(Encoder):
    def __init__(self, encoder: Encoder):
        super(CacheEncoder, self).__init__()
        self.encoder = encoder
        self.cache = self.get_cache_obj()

    def trainable(self) -> bool:
        return False

    def embedding_size(self) -> int:
        return self.encoder.embedding_size()

    @classmethod
    def get_cache_obj(cls) -> EmbeddingCacheAccessor:
        """
        Function that creates cache accessor object
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def key_extraction_fn(cls, record: Any) -> str:
        """
        Function that extracts cache key from record.
        :return: key for the given record
        """
        raise NotImplementedError()

    def get_collate_fn(self) -> CollateFnType:
        return CachedCollater(
            key_fn=self.__class__.key_extraction_fn,
            collate_fn=self.encoder.get_collate_fn(),
            cache_getter=self.__class__.get_cache_obj
        )

    def forward(self, batch: TensorInterchange) -> Tensor:
        records = batch['records']
        embeddings = self.encoder.forward(batch=records)

        self.cache.put_embeddings(batch['records_keys'], embeddings.clone().detach().cpu().numpy())

        cached_keys: List[str] = batch['cached_keys']

        if cached_keys:
            cached_embeddings = torch.tensor(self.cache.get_embeddings(keys=cached_keys), device=embeddings.device)

            # Shape: [batch_size x emb_size]
            embeddings = torch.cat([embeddings, cached_embeddings], dim=0)

        ordering = torch.tensor(
            batch['records_order'] + batch['cached_order'],
            device=embeddings.device
        )

        inverted_ordering = inverse_permutation(ordering)
        return embeddings[inverted_ordering]

    def save(self, output_path: str):
        encoder_data = save_class_import(self.encoder)
        encoder_path = os.path.join(output_path, 'encoder')
        os.makedirs(encoder_path, exist_ok=True)
        self.encoder.save(encoder_path)

        with open(os.path.join(output_path, 'config.json'), 'w') as f_out:
            json.dump({
                "encoder": encoder_data,
            }, f_out, indent=2)

    @classmethod
    def load(cls, input_path: str) -> 'Encoder':
        with open(os.path.join(input_path, 'config.json')) as f_in:
            config = json.load(f_in)

        encoder_params: dict = config["encoder"]
        encoder_path = os.path.join(input_path, 'encoder')
        encoder_class = restore_class(encoder_params)
        encoder = encoder_class.load(encoder_path)

        return cls(encoder=encoder)
