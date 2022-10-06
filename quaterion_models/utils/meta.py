from typing import Dict, List


def merge_meta(meta: Dict[str, list]) -> List[dict]:
    """Merge meta information from multiple encoders into one

    Combine meta from all encoders
    Example: Encoder 1 meta: `[{"a": 1}, {"a": 2}]`, Encoder 2 meta: `[{"b": 3}, {"b": 4}]`
    Result: `[{"a": 1, "b": 3}, {"a": 2, "b": 4}]`

    Args:
        meta: meta information to merge
    """
    aggregated = None
    for key, encoder_meta in meta.items():
        if aggregated is None:
            aggregated = encoder_meta
        else:
            for i in range(len(meta)):
                aggregated[i].update(encoder_meta[i])
    return aggregated
