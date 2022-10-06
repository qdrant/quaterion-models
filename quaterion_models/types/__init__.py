# TODO: Split them into separate files once we have more of them.
from typing import Any, Callable, Dict, List, Tuple, Union

from torch import Tensor

#:
TensorInterchange = Union[
    Tensor,
    Tuple[Tensor],
    List[Tensor],
    Dict[str, Tensor],
    Dict[str, dict],
    Any,
]
#:
CollateFnType = Callable[[List[Any]], TensorInterchange]
MetaExtractorFnType = Callable[[List[Any]], List[dict]]
