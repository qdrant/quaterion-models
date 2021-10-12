import torch

from quaterion_models.heads.softmax_head import SoftmaxEmbeddingsHead


def test_forward():
    head = SoftmaxEmbeddingsHead(output_groups=4, output_size_per_group=10, input_embedding_size=8)

    res = head.forward(
        torch.tensor([
            [0.1398, 0.4389, 0.1449, 0.7866, 0.2907, 0.5005, 0.3324, 0.5667],
            [0.1520, 0.7085, 0.4112, 0.7135, 0.7980, 0.1672, 0.2275, 0.6845],
            [0.7643, 0.8717, 0.4842, 0.3353, 0.3653, 0.2837, 0.2169, 0.6272],
            [0.8501, 0.5618, 0.3326, 0.1783, 0.0489, 0.3466, 0.1977, 0.4441],
            [0.2128, 0.9782, 0.0450, 0.3469, 0.9289, 0.7169, 0.9112, 0.8393]
        ])
    )

    assert res.shape == (5, 10 * 4)
