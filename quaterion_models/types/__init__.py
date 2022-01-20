# TODO: Split them into separate files once we have more of them.
TensorInterchange = Union[
    Tensor, Tuple[Tensor], List[Tensor], Dict[str, Tensor], Dict[str, dict], Any
]
CollateFnType = Callable[[List[Any]], TensorInterchange]
