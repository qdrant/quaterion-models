import torch

from quaterion_models.heads import EncoderHead


class EmptyHead(EncoderHead):
    def __init__(self, input_embedding_size: int):
        super(EmptyHead, self).__init__(input_embedding_size)

    def output_size(self) -> int:
        return self.input_embedding_size

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return input_vectors
