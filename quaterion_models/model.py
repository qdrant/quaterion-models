from typing import Dict

from torch import nn

from quaterion_models.encoder import Encoder


class MetricModel(nn.Module):

    def __init__(
            self,
            encoders: Dict[str, Encoder],
            head: nn.Module
    ):
        super(MetricModel, self).__init__()
        self.encoders = encoders
        # ToDo: register params

    def forward(self):
        pass

