import torch
from torch import nn


class EncoderHead(nn.Module):
    def __init__(self):
        super(EncoderHead, self).__init__()

    def output_size(self) -> int:
        raise NotImplementedError()

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
