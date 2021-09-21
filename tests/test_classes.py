from quaterion_models.utils.classes import save_class_import, restore_class


def test_restore_class():
    model_class = restore_class({
        "module": 'torch.nn',
        "class": 'Linear'
    })

    from torch.nn import Linear
    model: Linear = model_class(10, 10)

    assert model.out_features == 10


def test_save_class_import():
    from collections import Counter

    class_serialized = save_class_import(Counter())

    assert class_serialized['class'] == 'Counter'
    assert class_serialized['module'] == 'collections'
