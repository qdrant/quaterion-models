from __future__ import annotations

import json
import os
from functools import partial
from typing import Dict, Any, List

import torch
from torch import Tensor

from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange, CollateFnType
from quaterion_models.utils import save_class_import, restore_class, move_to_device


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class SwitchEncoder(Encoder):
    """Allows use alternative embeddings based on input data.

    For example, train shared embedding representation for images and texts.
    In this case image encoder should be used if input is an image and text encoder in other case.
    """

    @classmethod
    def encoder_selection(cls, record: Any) -> str:
        """Decide which encoder to use for given record.

        Args:
            record: input piece of data

        Returns:
            name of the related encoder
        """
        raise NotImplementedError()

    def __init__(self, options: Dict[str, Encoder]):
        super(SwitchEncoder, self).__init__()
        self.options = options

        embedding_sizes = set()
        for key, encoder in self.options.items():
            self.add_module(key, encoder)
            embedding_sizes.add(encoder.embedding_size)

        if len(embedding_sizes) != 1:
            raise RuntimeError(
                f"Alternative encoders have inconsistent output size: {embedding_sizes}"
            )

        self._embedding_size = list(embedding_sizes)[0]

    def disable_gradients_if_required(self):
        for encoder in self.options.values():
            encoder.disable_gradients_if_required()

    @property
    def trainable(self) -> bool:
        return any(encoder.trainable for encoder in self.options.values())

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @classmethod
    def switch_collate_fn(
        cls, batch: List[Any], encoder_collates: Dict[str, CollateFnType]
    ) -> TensorInterchange:
        switch_batches = dict((key, []) for key in encoder_collates.keys())
        switch_ordering = dict((key, []) for key in encoder_collates.keys())
        for original_id, record in enumerate(batch):
            record_encoder = cls.encoder_selection(record)
            switch_batches[record_encoder].append(record)
            switch_ordering[record_encoder].append(original_id)

        return {"ordering": switch_ordering, "batches": switch_batches}

    def get_collate_fn(self) -> CollateFnType:
        return partial(
            self.__class__.switch_collate_fn,
            encoder_collates=dict(
                (key, encoder.get_collate_fn()) for key, encoder in self.options.items()
            ),
        )

    def forward(self, batch: TensorInterchange) -> Tensor:
        switch_ordering: dict = batch["ordering"]
        switch_batches: dict = batch["batches"]
        embeddings = []
        ordering = []
        for key, batch in switch_batches.items():
            embeddings.append(self.options[key].forward(batch))
            ordering += switch_ordering[key]
        ordering_tensor: Tensor = inverse_permutation(torch.tensor(ordering))
        embeddings_tensor: Tensor = torch.cat(embeddings)
        ordering_tensor = move_to_device(ordering_tensor, embeddings_tensor.device)
        return embeddings_tensor[ordering_tensor]

    def save(self, output_path: str):
        encoders = {}
        for key, encoder in self.options.items():
            encoders[key] = save_class_import(encoder)
            encoder_path = os.path.join(output_path, key)
            os.makedirs(encoder_path, exist_ok=True)
            encoder.save(encoder_path)

        with open(os.path.join(output_path, "config.json"), "w") as f_out:
            json.dump(
                {
                    "encoders": encoders,
                },
                f_out,
                indent=2,
            )

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        with open(os.path.join(input_path, "config.json")) as f_in:
            config = json.load(f_in)

        encoders = {}
        encoders_params: dict = config["encoders"]
        for key, class_params in encoders_params.items():
            encoder_path = os.path.join(input_path, key)
            encoder_class = restore_class(class_params)
            encoders[key] = encoder_class.load(encoder_path)

        return SwitchEncoder(options=encoders)
