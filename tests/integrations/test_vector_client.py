"""Test TiDB Vector Search functionality."""
from __future__ import annotations

import os
from typing import List, Tuple
import sqlalchemy

import pytest

try:
    from tidb_vector.integrations import TiDBVectorClient, reset_vector_model  # noqa
    from tidb_vector.integrations.utils import (
        EmbeddingColumnMismatchError,
        check_table_existence,
    )  # noqa

    TABLE_NAME = "tidb_vector_test"
    CONNECTION_STRING = os.getenv("TEST_TiDB_CONNECTION_URL", "")

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
def node_embeddings() -> Tuple[list[str], list[str], list[list[float]], list[dict]]:
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
    metadatas = [
        {"page": 1, "category": "P1"},
        {"page": 2, "category": "P1"},
        {"page": 3, "category": "P2"},
    ]
    return (ids, documents, embeddings, metadatas)


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_basic_search(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test end to end tidb vector construction and search."""

    tidb_vs = TiDBVectorClient(
        table_name=TABLE_NAME,
        connection_string=CONNECTION_STRING,
        vector_dimension=ADA_TOKEN_COUNT,
        drop_existing_table=True,
    )

    # Add document to the tidb vector
    tidb_vs.insert(
        texts=node_embeddings[1], ids=node_embeddings[0], embeddings=node_embeddings[2]
    )

    # similarity search
    results = tidb_vs.query(text_to_embedding("foo"), k=3)
    tidb_vs.drop_table()

    # Check results
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == node_embeddings[0][0]


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_mismatch_vector_dimension(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test mismatch vector dimension."""

    reset_vector_model()
    try:
        tidb_vs = TiDBVectorClient(
            table_name=TABLE_NAME,
            connection_string=CONNECTION_STRING,
            vector_dimension=ADA_TOKEN_COUNT - 1,
            drop_existing_table=True,
        )
        tidb_vs.insert(
            texts=node_embeddings[1],
            ids=node_embeddings[0],
            embeddings=node_embeddings[2],
        )
        assert False, "[insert] mismatch vector dimension raised an error"
    except sqlalchemy.exc.StatementError:
        pass

    try:
        _ = TiDBVectorClient(
            table_name=TABLE_NAME,
            connection_string=CONNECTION_STRING,
            vector_dimension=ADA_TOKEN_COUNT,
        )
        assert (
            False
        ), "[vector client initialization] mismatch vector dimension raised an error"
    except EmbeddingColumnMismatchError:
        pass

    tidb_vs.drop_table()


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_various_distance_strategies(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test various distance strategies."""
    reset_vector_model()
    distance_strategies = ["l2", "cosine", "inner_product"]
    for distance_strategy in distance_strategies:
        tidb_vs = TiDBVectorClient(
            table_name=TABLE_NAME,
            connection_string=CONNECTION_STRING,
            distance_strategy=distance_strategy,  # type: ignore
            drop_existing_table=True,
        )

        # Add document to the tidb vector
        tidb_vs.insert(
            texts=node_embeddings[1],
            ids=node_embeddings[0],
            embeddings=node_embeddings[2],
        )

        # similarity search
        results = tidb_vs.query(text_to_embedding("foo"), k=3)
        tidb_vs.drop_table()

        # Check results
        assert len(results) == 3
        assert results[0].document == node_embeddings[1][0]
        assert results[0].id == node_embeddings[0][0]

    try:
        _ = TiDBVectorClient(
            table_name=TABLE_NAME,
            connection_string=CONNECTION_STRING,
            distance_strategy="error",  # type: ignore
            drop_existing_table=True,
        )
        assert False, "invalid distance strategy raised an error"
    except ValueError:
        pass

    tidb_vs.drop_table()


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_get_existing_table(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test get existing vector table."""

    reset_vector_model()
    # prepare a table
    tidb_vs = TiDBVectorClient(
        table_name=TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    tidb_vs.insert(
        texts=node_embeddings[1],
        ids=node_embeddings[0],
        embeddings=node_embeddings[2],
        metadatas=node_embeddings[3],
    )

    # try to get the existing vector table
    tidb_vs2 = TiDBVectorClient(
        table_name=TABLE_NAME,
        connection_string=CONNECTION_STRING,
    )
    results = tidb_vs2.query(text_to_embedding("bar"), k=3)
    # delete the table
    tidb_vs2.drop_table()
    # check the results
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][1]
    assert results[0].distance == 0.0
    assert results[0].id == node_embeddings[0][1]

    # it should fail if the table had been dropped
    try:
        results = tidb_vs.query(text_to_embedding("bar"), k=3)
        assert False, "dropped table testing raised an error"
    except Exception:
        pass

    # try to check table existence
    assert (
        check_table_existence(
            table_name=TABLE_NAME,
            connection_string=CONNECTION_STRING,
        )
        is False
    ), "non-existing table testing raised an error"


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_insert(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test insert function."""

    reset_vector_model()
    tidb_vs = TiDBVectorClient(
        table_name=TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    # Add document to the tidb vector
    ids = tidb_vs.insert(
        texts=node_embeddings[1],
        embeddings=node_embeddings[2],
        metadatas=node_embeddings[3],
    )

    results = tidb_vs.query(text_to_embedding("bar"), k=3)

    assert len(results) == 3
    assert results[0].document == node_embeddings[1][1]
    assert results[0].distance == 0.0
    assert results[0].id == ids[1]

    # Insert duplicate ids, it should raise an error
    try:
        _ = tidb_vs.insert(
            ids=ids,
            texts=node_embeddings[1],
            embeddings=node_embeddings[2],
            metadatas=node_embeddings[3],
        )
        tidb_vs.drop_table()
        assert False, "inserting to existing table raised an error"
    except sqlalchemy.exc.IntegrityError:
        tidb_vs.drop_table()
        pass


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_delete(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test delete function."""

    reset_vector_model()
    # prepare data
    tidb_vs = TiDBVectorClient(
        table_name=TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    ids = tidb_vs.insert(
        ids=node_embeddings[0],
        texts=node_embeddings[1],
        embeddings=node_embeddings[2],
        metadatas=node_embeddings[3],
    )

    results = tidb_vs.query(text_to_embedding("foo"), k=3)

    assert len(results) == 3
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == node_embeddings[0][0]

    # test delete by id

    # it should fail to delete first two documents conflicted with meta filter
    tidb_vs.delete([ids[1], ids[0]], filter={"category": "P2"})
    results = tidb_vs.query(text_to_embedding("foo"), k=4)
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == ids[0]

    # it should delete the first document just filtered by id
    tidb_vs.delete([ids[1], ids[0]])
    results = tidb_vs.query(text_to_embedding("foo"), k=4)
    assert len(results) == 1
    assert results[0].document == node_embeddings[1][2]
    assert results[0].distance == 0.004691842206844599
    assert results[0].id == node_embeddings[0][2]

    # insert the document back with different id
    ids = tidb_vs.insert(
        texts=node_embeddings[1],
        embeddings=node_embeddings[2],
        metadatas=node_embeddings[3],
    )

    results = tidb_vs.query(text_to_embedding("foo"), k=5)
    assert len(results) == 4
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == ids[0]

    # test delete first document by filter and ids
    tidb_vs.delete([ids[1], ids[0]], filter={"page": 1})
    results = tidb_vs.query(text_to_embedding("foo"), k=5)
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][1]
    assert results[1].document == node_embeddings[1][2]
    assert results[1].distance == results[2].distance

    # insert the document back with different id
    ids = tidb_vs.insert(
        texts=node_embeddings[1],
        embeddings=node_embeddings[2],
        metadatas=node_embeddings[3],
    )
    results = tidb_vs.query(text_to_embedding("foo"), k=10)
    assert len(results) == 6
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == ids[0]

    # test delete documents by filters
    tidb_vs.delete(filter={"category": "P1"})
    results = tidb_vs.query(text_to_embedding("foo"), k=10)
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][2]
    assert results[1].document == node_embeddings[1][2]
    assert results[0].distance == results[1].distance
    assert results[1].distance == results[2].distance

    # test delete non_extsting by filter
    tidb_vs.delete(filter={"category": "P1"})
    results = tidb_vs.query(text_to_embedding("foo"), k=10)
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][2]
    assert results[1].document == node_embeddings[1][2]
    assert results[0].distance == results[1].distance
    assert results[1].distance == results[2].distance

    # test delete non_extsting by ids
    tidb_vs.delete([ids[1], ids[0]])
    results = tidb_vs.query(text_to_embedding("foo"), k=10)
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][2]
    assert results[1].document == node_embeddings[1][2]
    assert results[0].distance == results[1].distance
    assert results[1].distance == results[2].distance

    # test delete non_extsting by filter and ids
    tidb_vs.delete([ids[1], ids[0]], filter={"category": "P1"})
    results = tidb_vs.query(text_to_embedding("foo"), k=10)
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][2]
    assert results[1].document == node_embeddings[1][2]
    assert results[0].distance == results[1].distance
    assert results[1].distance == results[2].distance

    tidb_vs.drop_table()


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_query(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test query function."""

    reset_vector_model()
    # prepare data
    tidb_vs = TiDBVectorClient(
        table_name=TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    ids = tidb_vs.insert(
        ids=node_embeddings[0],
        texts=node_embeddings[1],
        embeddings=node_embeddings[2],
        metadatas=node_embeddings[3],
    )

    results = tidb_vs.query(text_to_embedding("foo"), k=3)
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == ids[0]

    # test query by matched filter
    results = tidb_vs.query(text_to_embedding("foo"), k=3, filter={"category": "P1"})
    assert len(results) == 2
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == ids[0]

    # test query by another matched filter
    results = tidb_vs.query(text_to_embedding("foo"), k=3, filter={"category": "P2"})
    assert len(results) == 1
    assert results[0].document == node_embeddings[1][2]
    assert results[0].distance == 0.004691842206844599
    assert results[0].id == ids[2]

    # test query by unmatch filter
    results = tidb_vs.query(text_to_embedding("foo"), k=3, filter={"category": "P3"})
    assert len(results) == 0

    # test basic filter query
    results = tidb_vs.query(
        text_to_embedding("foo"), k=3, filter={"page": 2, "category": "P1"}
    )
    assert len(results) == 1
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(
        text_to_embedding("foo"), k=3, filter={"page": 1, "category": "P2"}
    )
    assert len(results) == 0

    results = tidb_vs.query(
        text_to_embedding("foo"), k=3, filter={"page": {"$gt": 1}, "category": "P1"}
    )
    assert len(results) == 1
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={"page": {"$gt": 1}, "category": {"$ne": "P2"}},
    )
    assert len(results) == 1
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={"page": {"$gt": 1}, "category": {"$ne": "P1"}},
    )
    assert len(results) == 1
    assert results[0].distance == 0.004691842206844599

    results = tidb_vs.query(
        text_to_embedding("foo"), k=3, filter={"page": {"$in": [2, 3]}}
    )
    assert len(results) == 2
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={"page": {"$in": [2, 3]}, "category": {"$ne": "P1"}},
    )
    assert len(results) == 1
    assert results[0].distance == 0.004691842206844599

    results = tidb_vs.query(
        text_to_embedding("foo"), k=3, filter={"page": {"$nin": [2, 3]}}
    )
    assert len(results) == 1
    assert results[0].distance == 0.0

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={"page": {"$nin": [2, 3]}, "category": {"$ne": "P1"}},
    )
    assert len(results) == 0

    results = tidb_vs.query(text_to_embedding("foo"), k=3, filter={"page": {"$gte": 2}})
    assert len(results) == 2
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(text_to_embedding("foo"), k=3, filter={"page": {"$lt": 4}})
    assert len(results) == 3
    assert results[0].distance == 0.0

    results = tidb_vs.query(text_to_embedding("baz"), k=3, filter={"page": {"$lte": 2}})
    assert len(results) == 2
    assert results[0].distance == 0.0005609046916807969

    results = tidb_vs.query(text_to_embedding("baz"), k=3, filter={"page": {"$eq": 2}})
    assert len(results) == 1
    assert results[0].distance == 0.0005609046916807969

    try:
        _ = tidb_vs.query(text_to_embedding("foo"), k=3, filter={"$and": [{"$gt": 1}]})
        tidb_vs.drop_table()
        assert False, "query with invalid filter raised an error"
    except ValueError:
        pass

    tidb_vs.drop_table()


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_complex_query(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test complex query function."""

    reset_vector_model()
    # prepare data
    tidb_vs = TiDBVectorClient(
        table_name=TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    ids = tidb_vs.insert(
        ids=node_embeddings[0],
        texts=node_embeddings[1],
        embeddings=node_embeddings[2],
        metadatas=node_embeddings[3],
    )

    results = tidb_vs.query(text_to_embedding("foo"), k=3)
    assert len(results) == 3
    assert results[0].document == node_embeddings[1][0]
    assert results[0].distance == 0.0
    assert results[0].id == ids[0]

    # test complex query
    results = tidb_vs.query(
        text_to_embedding("foo"), k=3, filter={"$and": [{"page": 1}]}
    )
    assert len(results) == 1
    assert results[0].distance == 0.0

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={"$and": [{"page": {"$gt": 1}}, {"category": "P1"}]},
    )
    assert len(results) == 1
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(
        text_to_embedding("foo"), k=3, filter={"$or": [{"page": 1}]}
    )
    assert len(results) == 1
    assert results[0].distance == 0.0

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={"$or": [{"page": {"$gt": 1}}, {"category": "P1"}]},
    )
    assert len(results) == 3
    assert results[0].distance == 0.0

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={
            "$and": [{"page": {"$gt": 1}}, {"category": "P1"}],
            "$or": [{"page": {"$gt": 1}}, {"category": "P1"}],
        },
    )
    assert len(results) == 1
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={"$and": [{"page": {"$gt": 1}}, {"category": "P1"}], "page": 1},
    )
    assert len(results) == 0

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={"$or": [{"page": {"$gt": 1}}, {"category": "P1"}], "page": 1},
    )
    assert len(results) == 1
    assert results[0].distance == 0.0

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={
            "$and": [{"page": {"$gt": 1}}, {"category": "P1"}],
            "page": 2,
            "$or": [{"page": {"$gt": 1}}, {"category": "P1"}],
        },
    )
    assert len(results) == 1
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={
            "$and": [
                {
                    "$and": [
                        {"page": {"$gt": 1}},
                        {"page": {"$lt": 3}},
                    ],
                    "category": "P2",
                }
            ]
        },
    )
    assert len(results) == 0

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={
            "$and": [
                {
                    "$and": [
                        {"page": {"$gt": 1}},
                        {"page": {"$lt": 3}},
                    ],
                    "$or": [
                        {"page": {"$gt": 2}},
                        {"category": {"$eq": "P1"}},
                    ],
                }
            ]
        },
    )
    assert len(results) == 1
    assert results[0].distance == 0.0022719614199674387

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={
            "$or": [
                {
                    "$and": [
                        {"page": {"$gt": 1}},
                        {"page": {"$lt": 3}},
                    ],
                    "category": "P2",
                },
                {
                    "category": "P2",
                },
            ]
        },
    )
    assert len(results) == 1
    assert results[0].distance == 0.004691842206844599

    results = tidb_vs.query(
        text_to_embedding("foo"),
        k=3,
        filter={
            "$or": [
                {
                    "$and": [
                        {"page": {"$gt": 1}},
                        {"page": {"$lt": 3}},
                    ],
                    "$or": [
                        {"page": {"$lt": 3}},
                        {"category": {"$eq": "P2"}},
                    ],
                },
                {
                    "category": "P2",
                },
            ]
        },
    )
    assert len(results) == 2
    assert results[0].distance == 0.0022719614199674387

    tidb_vs.drop_table()


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_execute(
    node_embeddings: Tuple[list[str], list[str], list[list[float]], list[dict]]
) -> None:
    """Test execute method with SELECT query."""

    reset_vector_model()
    tidb_vs = TiDBVectorClient(
        table_name=TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    # Insert data into the table
    tidb_vs.insert(
        texts=node_embeddings[1],
        ids=node_embeddings[0],
        embeddings=node_embeddings[2],
    )

    # Create a testing table
    result = tidb_vs.execute(
        "CREATE TABLE IF NOT EXISTS test_tidb_vector_execution_function (id VARCHAR(36), document VARCHAR(100), category VARCHAR(100))"
    )
    assert result["success"] is True
    assert result["error"] is None
    assert result["result"] == 0

    # Insert data into the testing table
    result = tidb_vs.execute(
        "INSERT INTO test_tidb_vector_execution_function (id, document, category) VALUES (:id, :document, :category)",
        {"id": "123", "document": "test", "category": "P1"},
    )
    assert result["success"] is True
    assert result["error"] is None
    assert result["result"] == 1

    # Execute a SELECT query
    result = tidb_vs.execute(
        "SELECT * FROM test_tidb_vector_execution_function WHERE category = :category",
        {"category": "P1"},
    )

    # Check the result
    assert result["success"] is True
    assert len(result["result"]) == 1
    assert result["error"] is None

    # Drop the testing table
    result = tidb_vs.execute("DROP TABLE test_tidb_vector_execution_function")
    assert result["success"] is True
    assert result["error"] is None
    assert result["result"] == 0

    assert (
        check_table_existence(
            table_name="test_tidb_vector_execution_function",
            connection_string=CONNECTION_STRING,
        )
        is False
    ), "Table test_tidb_vector_execution_function should not exist"

    # test non-existing table
    result = tidb_vs.execute("SELECT * FROM test_tidb_vector_execution_function")
    assert result["success"] is False
    assert result["error"] is not None

    tidb_vs.drop_table()
