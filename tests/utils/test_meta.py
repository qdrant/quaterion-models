from typing import Dict

from quaterion_models.utils.meta import merge_meta


def test_merge_meta():
    input_test: Dict[str, list] = {
        "encoder_a": [{"a": 1}, {"a": 2}],
        "encoder_b": [{"b": 3}, {"b": 4}],
        "encoder_c": [{"c": 5}, {"c": 6}],
    }
    expected = [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4, "c": 6}]
    actual = merge_meta(input_test)

    assert actual == expected
