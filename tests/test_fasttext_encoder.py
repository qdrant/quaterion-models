import os
import tempfile

from quaterion_models.encoders.fasttext_encoder import FasttextEncoder


def test_fasttext_encoder():
    from gensim.models import FastText

    demo_texts = [
        ["aaa", "bbb", "ccc", "ddd", "123"],
        ["aaa", "bbb", "ccc", "aaa", "123"],
        ["aaa", "bbb", "ccc", "bbb", "123"],
        ["aaa", "bbb", "ccc", "ccc", "123"],
        ["aaa", "bbb", "ccc", "123", "123"]
    ]

    model = FastText(
        vector_size=10,
        window=1,
        min_count=0,
        min_n=0,
        max_n=0,
        sg=1,
        bucket=1_000
    )
    epochs = 10
    model.build_vocab(demo_texts)
    model.train(demo_texts, epochs=epochs, total_examples=len(demo_texts))
    tempdir = tempfile.TemporaryDirectory()

    model_path = os.path.join(tempdir.name, 'fasttext.model')
    model.wv.save(model_path, separately=['vectors_ngrams', 'vectors', 'vectors_vocab'])

    encoder = FasttextEncoder(model_path=model_path, on_disk=False, aggregations=['avg', 'max'])

    assert encoder.embedding_size() == 20

    embeddings = encoder.forward([['aaa', '123'], ['aaa', 'ccc']])

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 20
