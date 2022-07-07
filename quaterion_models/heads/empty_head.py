import torch

from quaterion_models.heads.encoder_head import EncoderHead


class EmptyHead(EncoderHead):
    """Returns input embeddings without any modification"""

    def __init__(self, input_embedding_size: int, dropout: float = 0.0):
        super(EmptyHead, self).__init__(input_embedding_size, dropout=dropout)

    @property
    def output_size(self) -> int:
        return self.input_embedding_size

    def transform(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return input_vectors
