from functools import partial
from typing import Dict, List, Type, Callable

import torch
from torch import nn

from quaterion_models.encoder import Encoder, TensorInterchange
from quaterion_models.heads.encoder_head import EncoderHead


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
        return partial(MetricModel.collate_fn, encoders=dict(
            (key, encoder.__class__)
            for key, encoder in self.encoders.items()
        ))

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
