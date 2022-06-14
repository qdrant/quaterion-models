# Quaterion Models

`quaterion-models` is a part of [`Quaterion`](https://github.com/qdrant/quaterion), similarity learning framework.
It is kept as a separate package to make servable models lightweight and free from training dependencies.

It contains definition of base classes, used for model inference, as well as the collection of building blocks for building fine-tunable similarity learning models.
The documentation can be found [here](https://quaterion-models.qdrant.tech/).

If you are looking for the training-related part of Quaterion, please see the [main repository](https://github.com/qdrant/quaterion) instead.

## Install

```bash
pip install quaterion-models
```

It makes sense to install `quaterion-models` independent of the main framework if you already have trained model 
and only need to make inference.

## Load and inference

```python
from quaterion_models import SimilarityModel

model = SimilarityModel.load("./path/to/saved/model")

embeddings = model.encode([
    {"description": "this is an example input"},
    {"description": "you may have a different format"},
    {"description": "the output will be a numpy array"},
    {"description": "of size [batch_size, embedding_size]"},
])
```

## Content

* `SimilarityModel` - main class which contains encoder models with the head layer
* Base class for Encoders
* Base class and various implementations of the Head Layers
* Additional helper functions
