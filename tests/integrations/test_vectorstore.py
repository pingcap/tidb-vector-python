"""Test TiDB Vector Search functionality."""
from __future__ import annotations

import os
from typing import List, Tuple

import pytest

try:
    from tidb_vector.integrations import TiDBCollection  # noqa

    COLLECTION_NAME = "tidb_vector_index_test"
    CONNECTION_STRING = os.getenv("TEST_TiDB_VECTOR_URL", "")

    if CONNECTION_STRING == "":
        raise OSError("TEST_TiDB_URL environment variable is not set")

    tidb_available = True
except (OSError, ImportError):
    tidb_available = False

ADA_TOKEN_COUNT = 1536


def text_to_embedding(text: str) -> List[float]:
    """Convert text to a unique embedding using ASCII values."""
    ascii_values = [float(ord(char)) for char in text]
    # Pad or trim the list to make it of length ADA_TOKEN_COUNT
    return ascii_values[:ADA_TOKEN_COUNT] + [0.0] * (
        ADA_TOKEN_COUNT - len(ascii_values)
    )


@pytest.fixture(scope="session")
def node_embeddings() -> Tuple[list[str], list[str], list[list[float]]]:
    """Return a list of TextNodes with embeddings."""
    ids = [
        "f8e7dee2-63b6-42f1-8b60-2d46710c1971",
        "8dde1fbc-2522-4ca2-aedf-5dcb2966d1c6",
        "e4991349-d00b-485c-a481-f61695f2b5ae",
    ]
    embeddings = [
        text_to_embedding("foo"),
        text_to_embedding("bar"),
        text_to_embedding("baz"),
    ]
    documents = ["foo", "bar", "baz"]
    return (ids, documents, embeddings)


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_basic_search(
    node_embeddings: Tuple[list[str], list[str], list[list[float]]]
) -> None:
    """Test end to end construction and search."""

    tidbcol = TiDBCollection(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    # Add document to the tidb vector
    tidbcol.insert(
        texts=node_embeddings[1], ids=node_embeddings[0], embeddings=node_embeddings[2]
    )

    # similarity search
    results = tidbcol.query(text_to_embedding("foo"), k=3)

    tidbcol.drop_collection()
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == node_embeddings[0][0]
