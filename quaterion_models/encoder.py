from typing import List, Any, Union, Dict, Tuple, Callable

from torch import Tensor
from torch import nn

TensorInterchange = Union[Tensor, Tuple[Tensor], List[Tensor], Dict[str, Tensor], Dict[str, dict], Any]
CollateFnType = Callable[[List[Any]], TensorInterchange]


class Encoder(nn.Module):
    """
    Base class for encoder abstraction
    """

    def __init__(self):
        super(Encoder, self).__init__()

    def disable_gradients_if_required(self):
        """
        Disables gradients of the model if it is declared as not trainable
        :return:
        """
        if not self.trainable():
            for key, weights in self.named_parameters():
                weights.requires_grad = False

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

    def get_collate_fn(self) -> CollateFnType:
        """
        Provides function that converts raw data batch into suitable model input

        :return: Model input
        """
        raise NotImplementedError()

    def get_key_collate_fn(self) -> CollateFnType:
        """
        Provides function that produces keys for data batch in caching purposes

        :return: Collection of keys for cache
        """
        pass

    def forward(self, batch: TensorInterchange) -> Tensor:
        """
        Infer encoder - convert input batch to embeddings

        :param batch: processed batch
        :return: embeddings, shape: [batch_size x embedding_size]
        """
        raise NotImplementedError()

    def save(self, output_path: str):
        """
        Persist current state to the provided directory

        :param output_path:
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, input_path: str) -> 'Encoder':
        """
        Instantiate encoder from saved state.
        If no state required - just call `create` instead

        :param input_path:
        :return:
        """
        raise NotImplementedError()
