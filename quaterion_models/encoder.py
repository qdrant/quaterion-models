from typing import List, Any, Union, Dict, Tuple

from torch import Tensor
from torch import nn

TensorInterchange = Union[Tensor, Tuple[Tensor], Dict[str, Tensor], Dict[str, dict]]


class Encoder(nn.Module):
    """
    Base class for encoder abstraction
    """

    def __init__(self):
        super(Encoder, self).__init__()

    def trainable(self) -> bool:
        """
        Defines if encoder is trainable. This flag affects caching and checkpoint saving of the encoder.
        :return: bool
        """
        raise NotImplementedError()

    def embedding_size(self) -> int:
        """
        :return: Size of resulting embedding
        """
        raise NotImplementedError()

    @classmethod
    def collate(cls, batch: List[Any]) -> TensorInterchange:
        """
        Convert raw data batch into suitable model input

        :param batch: List of any input data
        :return: Model input
        """
        raise NotImplementedError()

    def forward(self, batch: TensorInterchange) -> Tensor:
        """
        Infer encoder - convert input batch to embeddings

        :param batch: processed batch
        :return: embeddings, shape: [batch_size x embedding_size]
        """
        raise NotImplementedError()
