import os
from dotenv import load_dotenv
import pytest
from functools import partial
from utils import sentence_transformer_embedding_function, Vector, Vectors
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer

load_dotenv()


class VectorModel(BaseModel):
    vector: Vector


class VectorsModel(BaseModel):
    vectors: Vectors


@pytest.fixture(scope='module')
def embed_model():
    return SentenceTransformer(os.environ.get('SENTENCE_TRANSFORMERS_MODEL'), trust_remote_code=True)


def test_sentence_transformer_embedding_function_return_shape(embed_model: SentenceTransformer):
    embed_model_dim = embed_model.get_sentence_embedding_dimension()

    assert embed_model.encode(["Hello, world!"]).shape == (1, embed_model_dim)
    assert embed_model.encode(["Hello, world!", "hi"]).shape == (2, embed_model_dim)
    assert embed_model.encode("Hello, World!").shape == (embed_model_dim,)


def test_embedding_function(embed_model: SentenceTransformer):
    embedding_function = partial(sentence_transformer_embedding_function, embed_model)
    try:
        vector = embedding_function(sentences="Hello, world!")
        VectorModel(vector=vector)
    except ValidationError:
        assert False
    try:
        vectors = embedding_function(sentences=["Hello, world!"])
        VectorsModel(vectors=vectors)
    except ValidationError:
        assert False
    try:
        vectors = embedding_function(sentences=["Hello, world!", "hi"])
        VectorsModel(vectors=vectors)
    except ValidationError:
        assert False
