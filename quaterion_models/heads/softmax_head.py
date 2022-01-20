import torch
from torch.nn import Linear

from quaterion_models.heads import EncoderHead


class SoftmaxEmbeddingsHead(EncoderHead):
    def output_size(self) -> int:
        return self.output_size_per_group * self.output_group

    def __init__(
        self,
        output_groups: int,
        output_size_per_group: int,
        input_embedding_size: int,
        **kwargs
    ):

        super(SoftmaxEmbeddingsHead, self).__init__(input_embedding_size, **kwargs)

        self.output_groups = output_groups
        self.output_size_per_group = output_size_per_group
        self.projectors = []

        self.projection_layer = Linear(
            self.input_embedding_size, self.output_size_per_group * self.output_groups
        )

    def forward(self, input_vectors: torch.Tensor):
        """

        :param input_vectors: shape: [batch_size, ..., input_dim]
        :return: shape [batch_size, ..., self.output_size_per_group * self.output_groups]
        """

        # shape: [batch_size, ..., self.output_size_per_group * self.output_groups]
        projection = self.projection_layer(input_vectors)

        init_shape = projection.shape
        groups_shape = list(init_shape)
        groups_shape[-1] = self.output_groups
        groups_shape.append(-1)

        # shape: [batch_size, ..., self.output_groups, self.output_size_per_group]
        grouped_projection = torch.softmax(projection.view(*groups_shape), dim=-1)

        return grouped_projection.view(init_shape)
