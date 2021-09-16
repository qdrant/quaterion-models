from functools import partial
from typing import Dict, List, Type, Callable, Any, Union

import numpy as np
import torch
from torch import nn

from quaterion_models.encoder import Encoder, TensorInterchange
from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.utils.tensors import move_to_device


class MetricModel(nn.Module):

    def __init__(
            self,
            encoders: Dict[str, Encoder],
            head: EncoderHead
    ):
        super(MetricModel, self).__init__()
        self.encoders = encoders
        self.head = head

    @classmethod
    def collate_fn(cls, batch: List[dict], encoders: Dict[str, Type[Encoder]]) -> TensorInterchange:
        """
        Construct batches for all encoders

        :param batch:
        :param encoders:
        :return:
        """
        result = dict(
            (key, encoder.collate(batch))
            for key, encoder in encoders.items()
        )
        return result

    def get_collate_fn(self) -> Callable:
        """
        Construct a function to convert input data into neural network inputs

        :return: neural network inputs
        """
        return partial(MetricModel.collate_fn, encoders=dict(
            (key, encoder.__class__)
            for key, encoder in self.encoders.items()
        ))

    def encode(self,
               inputs: Union[List[Any], Any],
               batch_size=32,
               to_numpy=True
               ) -> Union[torch.Tensor, np.ndarray]:

        """
        Encode data in batches

        :param inputs: list of input data to encode
        :param batch_size:
        :param to_numpy:
        :return: Numpy array or torch.Tensor of shape [input_size x embedding_size]
        """
        self.eval()
        device = next(self.parameters()).device
        collate_fn = self.get_collate_fn()

        input_was_list = True
        if not isinstance(inputs, list):
            input_was_list = False
            inputs = [inputs]

        all_embeddings = []

        for start_index in range(0, len(inputs), batch_size):
            input_batch = inputs[start_index:start_index + batch_size]
            features = collate_fn(input_batch)
            features = move_to_device(features, device)

            with torch.no_grad():
                embeddings = self.model.forward(features)
                embeddings = embeddings.detach()
                if to_numpy:
                    embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

        if to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        if not input_was_list:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def forward(self, batch):
        embeddings = [
            (key, encoder.forward(batch[key]))
            for key, encoder in self.encoders.items()
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
