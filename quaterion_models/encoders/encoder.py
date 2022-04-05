from __future__ import annotations

from torch import Tensor
from torch import nn
from torch.utils.data.dataloader import default_collate

from quaterion_models.types import TensorInterchange, CollateFnType


class Encoder(nn.Module):
    """Base class for encoder abstraction"""

    def __init__(self):
        super(Encoder, self).__init__()

    def disable_gradients_if_required(self):
        """Disables gradients of the model if it is declared as not trainable"""

        if not self.trainable:
            for _, weights in self.named_parameters():
                weights.requires_grad = False

    @property
    def trainable(self) -> bool:
        """Defines if encoder is trainable.

        This flag affects caching and checkpoint saving of the encoder.
        """
        raise NotImplementedError()

    @property
    def embedding_size(self) -> int:
        """Size of resulting embedding"""
        raise NotImplementedError()

    def get_collate_fn(self) -> CollateFnType:
        """Provides function that converts raw data batch into suitable model input

        Returns:
             :const:`~quaterion_models.types.CollateFnType`: model's collate function
        """
        return default_collate

    def forward(self, batch: TensorInterchange) -> Tensor:
        """Infer encoder - convert input batch to embeddings

        Args:
            batch: processed batch
        Returns:
            embeddings: shape: (batch_size, embedding_size)
        """
        raise NotImplementedError()

    def save(self, output_path: str):
        """Persist current state to the provided directory

        Args:
            output_path: path to save model

        """
        raise NotImplementedError()

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        """Instantiate encoder from saved state.

        If no state required - just call `create` instead

        Args:
            input_path: path to load from

        Returns:
            :class:`~quaterion_models.encoders.encoder.Encoder`: loaded encoder
        """
        raise NotImplementedError()
