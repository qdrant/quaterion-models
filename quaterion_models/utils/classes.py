import importlib
import logging
from typing import Any


def save_class_import(obj: Any) -> dict:
    """
    Serializes information about object class
    :param obj:
    :return: serializable class info
    """
    if obj.__module__ == "__main__":
        logging.warning(
            f"Class {obj.__class__.__qualname__} is defined in a same file as training loop."
            f" It won't be possible to load it properly later."
        )

    return {"module": obj.__module__, "class": obj.__class__.__qualname__}


def restore_class(data: dict) -> Any:
    """
    :param data: name of module and class
    :return: Class
    """
    module = data["module"]
    class_name = data["class"]
    module = importlib.import_module(module)
    return getattr(module, class_name)
