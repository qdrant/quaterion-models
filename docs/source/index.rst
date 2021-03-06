Welcome to quaterion-models's documentation!
============================================


``quaterion-models`` is a part of
`Quaterion <https://quaterion.qdrant.tech/>`_, similarity
learning framework. It is kept as a separate package to make servable
models lightweight and free from training dependencies.

It contains definition of base classes, used for model inference, as
well as the collection of building blocks for building fine-tunable
similarity learning models.

If you are looking for the training-related part of Quaterion, please
see the `main repository <https://github.com/qdrant/quaterion>`_
instead.

Install
-------

.. code:: bash

   pip install quaterion-models

It makes sense to install ``quaterion-models`` independent of the main
framework if you already have trained model and only need to make
inference.

Load and inference
------------------

.. code:: python

   from quaterion_models import SimilarityModel

   model = SimilarityModel.load("./path/to/saved/model")

   embeddings = model.encode([
       {"description": "this is an example input"},
       {"description": "you may have a different format"},
       {"description": "the output will be a numpy array"},
       {"description": "of size [batch_size, embedding_size]"},
   ])

Content
-------

.. toctree::
   :maxdepth: 1

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
