from __future__ import annotations

import json
import os
from typing import List, Any, Union

import gensim
import numpy as np
import torch
from gensim.models import FastText, KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors
from torch import Tensor

from quaterion_models.encoders import Encoder
from quaterion_models.types import CollateFnType


def load_fasttext_model(path: str) -> Union[FastText, KeyedVectors]:
    """Load fasttext model in a universal way

    Try to find possible way of loading FastText model and load it

    Args:
        path: path to FastText model or vectors

    Returns:
        :class:`~gensim.models.fasttext.FastText` or :class:`~gensim.models.KeyedVectors`:
        loaded model
    """
    try:
        model = FastText.load(path).wv
    except Exception:
        try:
            model = FastText.load_fasttext_format(path).wv
        except Exception:
            model = gensim.models.KeyedVectors.load(path)

    return model


class FasttextEncoder(Encoder):
    """Creates a fasttext encoder, which generates vector for a list of tokens based in given
    fasttext model

    Args:
        model_path: Path to model to load
        on_disk: If True - use mmap to keep embeddings out of RAM
        aggregations: What types of aggregations to use to combine multiple vectors into one. If
            multiple aggregations are specified - concatenation of all of them will be used as a
            result.

    """

    aggregation_options = ["min", "max", "avg"]

    def __init__(self, model_path: str, on_disk: bool, aggregations: List[str] = None):
        super(FasttextEncoder, self).__init__()

        # workaround tensor to keep information about required model device
        self._device_tensor = torch.nn.Parameter(torch.zeros(1))

        if aggregations is None:
            aggregations = ["avg"]
        self.aggregations = aggregations
        self.on_disk = on_disk

        # noinspection PyTypeChecker
        self.model: FastTextKeyedVectors = gensim.models.KeyedVectors.load(
            model_path, mmap="r" if self.on_disk else None
        )

    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return self.model.vector_size * len(self.aggregations)

    @classmethod
    def get_tokens(cls, batch: List[Any]) -> List[List[str]]:
        raise NotImplementedError()

    def get_collate_fn(self) -> CollateFnType:
        return self.__class__.get_tokens

    @classmethod
    def aggregate(cls, embeddings: Tensor, operation: str) -> Tensor:
        """Apply aggregation operation to embeddings along the first dimension

        Args:
            embeddings: embeddings to aggregate
            operation: one of :attr:`aggregation_options`

        Returns:
            Tensor: aggregated embeddings
        """
        if operation == "avg":
            return torch.mean(embeddings, dim=0)
        if operation == "max":
            return torch.max(embeddings, dim=0).values
        if operation == "min":
            return torch.min(embeddings, dim=0).values

        raise RuntimeError(f"Unknown operation: {operation}")

    def forward(self, batch: List[List[str]]) -> Tensor:
        embeddings = []
        for record in batch:
            token_vectors = [self.model.get_vector(token) for token in record]
            if token_vectors:
                record_vectors = np.stack(token_vectors)
            else:
                record_vectors = np.zeros((1, self.model.vector_size))
            token_tensor = torch.tensor(
                record_vectors, device=self._device_tensor.device
            )
            record_embedding = torch.cat(
                [
                    self.aggregate(token_tensor, operation)
                    for operation in self.aggregations
                ]
            )
            embeddings.append(record_embedding)

        return torch.stack(embeddings)

    def save(self, output_path: str):
        model_path = os.path.join(output_path, "fasttext.model")
        self.model.save(
            model_path, separately=["vectors_ngrams", "vectors", "vectors_vocab"]
        )
        with open(os.path.join(output_path, "config.json"), "w") as f_out:
            json.dump(
                {
                    "on_disk": self.on_disk,
                    "aggregations": self.aggregations,
                },
                f_out,
                indent=2,
            )

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        model_path = os.path.join(input_path, "fasttext.model")
        with open(os.path.join(input_path, "config.json")) as f_in:
            config = json.load(f_in)

        return cls(model_path=model_path, **config)
