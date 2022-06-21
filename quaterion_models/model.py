from __future__ import annotations

import json
import os
from functools import partial
from typing import Dict, List, Type, Callable, Any, Union

import numpy as np
import torch
from torch import nn

from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange, CollateFnType
from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.utils.classes import save_class_import, restore_class
from quaterion_models.utils.tensors import move_to_device


DEFAULT_ENCODER_KEY = "default"


class SimilarityModel(nn.Module):
    """Main class which contains encoder models with the head layer."""

    def __init__(self, encoders: Union[Encoder, Dict[str, Encoder]], head: EncoderHead):
        super().__init__()

        if not isinstance(encoders, dict):
            self.encoders: Dict[str, Encoder] = {DEFAULT_ENCODER_KEY: encoders}
        else:
            self.encoders: Dict[str, Encoder] = encoders

        for key, encoder in self.encoders.items():
            encoder.disable_gradients_if_required()
            self.add_module(key, encoder)

        self.head = head

    @classmethod
    def collate_fn(
        cls, batch: List[dict], encoders_collate_fns: Dict[str, CollateFnType]
    ) -> TensorInterchange:
        """Construct batches for all encoders

        Args:
            batch:
            encoders_collate_fns: Dict (or single) of collate functions associated with encoders

        """
        result = dict(
            (key, collate_fn(batch)) for key, collate_fn in encoders_collate_fns.items()
        )
        return result

    @classmethod
    def get_encoders_output_size(cls, encoders: Union[Encoder, Dict[str, Encoder]]):
        """Calculate total output size of given encoders

        Args:
            encoders:

        """
        encoders = encoders.values() if isinstance(encoders, dict) else [encoders]
        total_size = 0
        for encoder in encoders:
            total_size += encoder.embedding_size
        return total_size

    def train(self, mode: bool = True):
        super().train(mode)

    def get_collate_fn(self) -> Callable:
        """Construct a function to convert input data into neural network inputs

        Returns:
            neural network inputs
        """
        return partial(
            SimilarityModel.collate_fn,
            encoders_collate_fns=dict(
                (key, encoder.get_collate_fn())
                for key, encoder in self.encoders.items()
            ),
        )

    # -------------------------------------------
    # ---------- Inference methods --------------
    # -------------------------------------------

    def encode(
        self, inputs: Union[List[Any], Any], batch_size=32, to_numpy=True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encode data in batches

        Args:
            inputs: list of input data to encode
            batch_size:
            to_numpy:

        Returns:
            Numpy array or torch.Tensor of shape (input_size, embedding_size)
        """
        self.eval()
        device = next(self.parameters(), torch.tensor(0)).device
        collate_fn = self.get_collate_fn()

        input_was_list = True
        if not isinstance(inputs, list):
            input_was_list = False
            inputs = [inputs]

        all_embeddings = []

        for start_index in range(0, len(inputs), batch_size):
            input_batch = [
                inputs[i]
                for i in range(start_index, min(len(inputs), start_index + batch_size))
            ]
            features = collate_fn(input_batch)
            features = move_to_device(features, device)

            with torch.no_grad():
                embeddings = self.forward(features)
                embeddings = embeddings.detach()
                if to_numpy:
                    embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

        if to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        if not input_was_list:
            all_embeddings = all_embeddings.squeeze()

        if to_numpy:
            all_embeddings = np.atleast_2d(all_embeddings)
        else:
            all_embeddings = torch.atleast_2d(all_embeddings)

        return all_embeddings

    def forward(self, batch):
        embeddings = [
            (key, encoder.forward(batch[key])) for key, encoder in self.encoders.items()
        ]
        # Order embeddings by key name, to ensure reproduction
        embeddings = sorted(embeddings, key=lambda x: x[0])

        # Only embedding tensors of shape [batch_size x encoder_output_size]
        embedding_tensors = [embedding[1] for embedding in embeddings]

        # Shape: [batch_size x sum( encoders_emb_sizes )]
        joined_embeddings = torch.cat(embedding_tensors, dim=1)

        # Shape: [batch_size x output_emb_size]
        result_embedding = self.head(joined_embeddings)

        return result_embedding

    # -------------------------------------------
    # ---------- Persistence methods ------------
    # -------------------------------------------

    @classmethod
    def _get_head_path(cls, directory: str):
        return os.path.join(directory, "head")

    @classmethod
    def _get_encoders_path(cls, directory: str):
        return os.path.join(directory, "encoders")

    def save(self, output_path: str):
        head_path = self._get_head_path(output_path)
        os.makedirs(head_path, exist_ok=True)
        self.head.save(head_path)

        head_config = save_class_import(self.head)

        encoders_path = self._get_encoders_path(output_path)
        os.makedirs(encoders_path, exist_ok=True)

        encoders_config = []

        for encoder_key, encoder in self.encoders.items():
            encoder_path = os.path.join(encoders_path, encoder_key)
            os.mkdir(encoder_path)
            encoder.save(encoder_path)
            encoders_config.append({"key": encoder_key, **save_class_import(encoder)})

        with open(os.path.join(output_path, "config.json"), "w") as f_out:
            json.dump(
                {"encoders": encoders_config, "head": head_config}, f_out, indent=2
            )

    @classmethod
    def load(cls, input_path: str) -> SimilarityModel:
        with open(os.path.join(input_path, "config.json")) as f_in:
            config = json.load(f_in)

        head_config = config["head"]
        head_class: Type[EncoderHead] = restore_class(head_config)
        head_path = cls._get_head_path(input_path)
        head = head_class.load(head_path)

        encoders: Union[Encoder, Dict[str, Encoder]] = {}
        encoders_path = cls._get_encoders_path(input_path)
        encoders_config = config["encoders"]

        for encoder_params in encoders_config:
            encoder_key = encoder_params["key"]
            encoder_class = restore_class(encoder_params)
            encoders[encoder_key] = encoder_class.load(
                os.path.join(encoders_path, encoder_key)
            )

        return cls(head=head, encoders=encoders)


# In this framework, the terms Metric Learning and Similarity Learning are considered synonymous.
# However, the word "Metric" overlaps with other concepts in model training.
# In addition, the semantics of the word "Similarity" are simpler.
# It better reflects the basic idea of this training approach.
# That's why we prefer to use Similarity over Metric.
MetricModel = SimilarityModel
