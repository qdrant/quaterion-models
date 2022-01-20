# TODO: Split them into separate files once we have more of them.
from typing import Union, Tuple, List, Dict, Any, Callable

from torch import Tensor

TensorInterchange = Union[
    Tensor,
    Tuple[Tensor],
    List[Tensor],
    Dict[str, Tensor],
    Dict[str, dict],
    Any,
]
CollateFnType = Callable[[List[Any]], TensorInterchange]
