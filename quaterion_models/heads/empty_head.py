import torch

from quaterion_models.heads.encoder_head import EncoderHead


class EmptyHead(EncoderHead):
    def __init__(self, vector_size: int):
        super(EmptyHead, self).__init__()
        self.vector_size = vector_size

    def output_size(self) -> int:
        return self.vector_size

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return input_vectors
