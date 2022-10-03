from typing import List, Dict


def merge_meta(meta: Dict[str, list]) -> List[dict]:
    """Merge meta information from multiple encoders into one

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
