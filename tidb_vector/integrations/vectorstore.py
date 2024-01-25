import contextlib
from dataclasses import dataclass
import logging
import enum
import uuid
from typing import Any, Dict, Generator, Iterable, List, Optional

import sqlalchemy
from sqlalchemy.orm import Session
from tidb_vector.integrations.model import (
    Base,
    get_simpledoc_table_model,
    get_vector_table_model,
)

logger = logging.getLogger()


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"


_classes: Any = None


def _create_vector_model(vector_table_name: str, content_table_name: str):
    """Create a vector model class."""

    global _classes
    if _classes is not None:
        return _classes

    _classes = (
        get_simpledoc_table_model(content_table_name),
        get_vector_table_model(vector_table_name),
    )

    return _classes


@dataclass
class QueryResult:
    id: str
    document: str
    metadata: dict
    distance: float


class VectorStore:
    def __init__(
        self,
        connection_string: str,
        table_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        drop_existing_table: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a vector store in a specified table within a TiDB database.

        Args:
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@127.0.0.1:4000/test".
            table_name (str): The name of the table used to store the vectors.
            distance_strategy: The strategy used for similarity search,
                defaults to "cosine", valid values: "l2", "cosine".
            engine_args (Optional[Dict]): Additional arguments for the database engine,
                defaults to None.
            drop_existing_table: Delete the table before creating a new one,
                defaults to False.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(**kwargs)
        self.connection_string = connection_string
        self._distance_strategy = distance_strategy
        self._engine_args = engine_args or {}
        self._drop_existing_table = drop_existing_table
        self._bind = self._create_engine()
        self._content_model, self._vector_model = _create_vector_model(
            table_name, f"{table_name}_simpledocs"
        )
        _ = self.distance_strategy  # check if distance strategy is valid
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """
        If the `self._pre_delete_table` flag is set,
        the existing table will be dropped before creating a new one.
        """
        if self._drop_existing_table:
            self.drop_table()
        with Session(self._bind) as session, session.begin():
            Base.metadata.create_all(session.get_bind())
            # wait for tidb support vector index

    def drop_table(self) -> None:
        """Drops the table if it exists."""
        with Session(self._bind) as session, session.begin():
            Base.metadata.drop_all(session.get_bind())

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        """Create a sqlalchemy engine."""
        return sqlalchemy.create_engine(url=self.connection_string, **self._engine_args)

    def __del__(self) -> None:
        """Close the connection when the program is closed"""
        if isinstance(self._bind, sqlalchemy.engine.Connection):
            self._bind.close()

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session."""
        yield Session(self._bind)

    @property
    def distance_strategy(self) -> Any:
        """
        Returns the distance function based on the current distance strategy value.
        """
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._vector_model.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self._vector_model.embedding.cosine_distance
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    @classmethod
    def get_vectorstore(
        cls,
        connection_string: str,
        table_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """
        Create a VectorStore instance from an existing table in the TiDB database.

        Args:
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@127.0.0.1:4000/test".
            table_name (str, optional): The name of the table used to store vector data.
            distance_strategy: The distance strategy used for similarity search,
                allowed strategies: "l2", "cosine".
            engine_args: Additional arguments for the underlying database engine,
                defaults to None.
            **kwargs (Any): Additional keyword arguments.
        Returns:
            VectorStore: The VectorStore instance.

        Raises:
            NoSuchTableError: If the specified table does not exist in the TiDB.
        """

        engine = sqlalchemy.create_engine(connection_string, **(engine_args or {}))

        try:
            # check if the table exists
            table_query = sqlalchemy.sql.text(
                "SELECT 1 FROM information_schema.tables WHERE table_name = :table_name"
            )
            with engine.connect() as connection:
                if (
                    connection.execute(
                        table_query,
                        {"table_name": table_name},
                    ).fetchone()
                    is None
                ):
                    raise sqlalchemy.exc.NoSuchTableError(
                        f"The table '{table_name}' does not exist in the database."
                    )

            return cls(
                connection_string=connection_string,
                table_name=table_name,
                distance_strategy=distance_strategy,
                engine_args=engine_args,
                **kwargs,
            )
        finally:
            # Close the engine after querying the tale
            engine.dispose()

    def insert(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to TiDB Vector.

        Args:
            texts (Iterable[str]): The texts to be added.
            metadatas (Optional[List[dict]]): The metadata associated with each text,
                Defaults to None.
            ids (Optional[List[str]]): The IDs to be assigned to each text,
                Defaults to None, will be generated if not provided.

        Returns:
            List[str]: The IDs assigned to the added texts.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]

        with Session(self._bind) as session:
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                content_id = self._content_model.insert(
                    session, text, metadata, commit_transaction=False
                )
                embeded_doc = self._vector_model(
                    id=id,
                    content_id=content_id,
                    ref_content_table=self._content_model.__tablename__,
                    embedding=embedding,
                    meta=metadata,
                )
                session.add(embeded_doc)
            session.commit()

        return ids

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete vector data from the TiDB vector.

        Args:
            ids (Optional[List[str]]): A list of vector IDs to delete.
            **kwargs: Additional keyword arguments.
        """
        filter_by = self._build_filter_clause(filter)
        with Session(self._bind) as session:
            if ids is not None:
                filter_by = sqlalchemy.and_(self._vector_model.id.in_(ids), filter_by)

            content_ids_query = (
                session.query(self._vector_model.content_id)
                .filter(filter_by)
                .distinct()
            )
            content_ids = [row.content_id for row in content_ids_query.all()]

            if not content_ids or len(content_ids) == 0:
                return

            session.query(self._vector_model).filter(
                self._vector_model.content_id.in_(content_ids)
            ).delete(synchronize_session=False)
            self._content_model.delete(session, content_ids, commit_transaction=False)

            session.commit()

    def query(
        self,
        query_vector: List[float],
        k: int = 5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[QueryResult]:
        """
        Perform a similarity search with score based on the given query.

        Args:
            query (str): The query string.
            k (int, optional): The number of results to return. Defaults to 5.
            filter (dict, optional): A filter to apply to the search results.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of tuples containing relevant documents and their similarity scores.
        """

        # Step 1: Get content IDs and additional information from vector search
        vector_search_results = self._vector_search(query_vector, k, filter)

        # Step 2: Fetch content details based on content IDs
        content_ids = [result.content_id for result in vector_search_results]
        if not content_ids and len(content_ids) == 0:
            return []

        contents = {}
        with Session(self._bind) as session:
            content_results = self._content_model.get_by_ids(session, content_ids)
            # Create a dictionary to map content IDs to their content for easy lookup
            contents = {content.content_id: content for content in content_results}

        # Step 3: Combine vector search results with content details
        query_results = []
        for result in vector_search_results:
            content = contents.get(result.content_id)
            if content:
                query_results.append(
                    QueryResult(
                        document=content.content,
                        metadata=result.meta,
                        id=result.id,
                        distance=result.distance,
                    )
                )

        return query_results

    def _vector_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """vector search from table."""

        filter_by = self._build_filter_clause(filter)
        with Session(self._bind) as session:
            results: List[Any] = (
                session.query(
                    self._vector_model.id,
                    self._vector_model.meta,
                    self._vector_model.content_id,
                    self.distance_strategy(query_embedding).label("distance"),
                )
                .filter(filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .limit(k)
                .all()
            )
        return results

    def _build_filter_clause(self, filters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Builds the filter clause for querying based on the provided filters.

        Args:
            filter (Dict[str, str]): The filter conditions to apply.

        Returns:
            Any: The filter clause to be used in the query on TiDB.
        """

        filter_by = sqlalchemy.true()
        if filters is not None:
            filter_clauses = []

            for key, value in filters.items():
                if key.lower() == "$and":
                    and_clauses = [
                        self._build_filter_clause(condition)
                        for condition in value
                        if isinstance(condition, dict) and condition is not None
                    ]
                    filter_by_metadata = sqlalchemy.and_(*and_clauses)
                    filter_clauses.append(filter_by_metadata)
                elif key.lower() == "$or":
                    or_clauses = [
                        self._build_filter_clause(condition)
                        for condition in value
                        if isinstance(condition, dict) and condition is not None
                    ]
                    filter_by_metadata = sqlalchemy.or_(*or_clauses)
                    filter_clauses.append(filter_by_metadata)
                elif key.lower() in [
                    "$in",
                    "$nin",
                    "$gt",
                    "$gte",
                    "$lt",
                    "$lte",
                    "$eq",
                    "$ne",
                ]:
                    raise ValueError(
                        f"Got unexpected filter expression: {filter}. "
                        f"Operator {key} must be followed by a meta key. "
                    )
                elif isinstance(value, dict):
                    filter_by_metadata = self._create_filter_clause(key, value)

                    if filter_by_metadata is not None:
                        filter_clauses.append(filter_by_metadata)
                else:
                    filter_by_metadata = (
                        sqlalchemy.func.json_extract(
                            self._vector_model.meta, f"$.{key}"
                        )
                        == value
                    )
                    filter_clauses.append(filter_by_metadata)

            filter_by = sqlalchemy.and_(filter_by, *filter_clauses)
        return filter_by

    def _create_filter_clause(self, key, value):
        """
        Create a filter clause based on the provided key-value pair.

        Args:
            key (str): How to filter the value
            value (dict): The value to filter with.

        Returns:
            sqlalchemy.sql.elements.BinaryExpression: The filter clause.

        Raises:
            None

        """

        IN, NIN, GT, GTE, LT, LTE, EQ, NE = (
            "$in",
            "$nin",
            "$gt",
            "$gte",
            "$lt",
            "$lte",
            "$eq",
            "$ne",
        )

        json_key = sqlalchemy.func.json_extract(self._vector_model.meta, f"$.{key}")
        value_case_insensitive = {k.lower(): v for k, v in value.items()}

        if IN in map(str.lower, value):
            filter_by_metadata = json_key.in_(value_case_insensitive[IN])
        elif NIN in map(str.lower, value):
            filter_by_metadata = ~json_key.in_(value_case_insensitive[NIN])
        elif GT in map(str.lower, value):
            filter_by_metadata = json_key > value_case_insensitive[GT]
        elif GTE in map(str.lower, value):
            filter_by_metadata = json_key >= value_case_insensitive[GTE]
        elif LT in map(str.lower, value):
            filter_by_metadata = json_key < value_case_insensitive[LT]
        elif LTE in map(str.lower, value):
            filter_by_metadata = json_key <= value_case_insensitive[LTE]
        elif NE in map(str.lower, value):
            filter_by_metadata = json_key != value_case_insensitive[NE]
        elif EQ in map(str.lower, value):
            filter_by_metadata = json_key == value_case_insensitive[EQ]
        else:
            logger.warning(
                f"Unsupported filter operator: {value}. Consider using "
                "one of $in, $nin, $gt, $gte, $lt, $lte, $eq, $ne, $or, $and."
            )
            filter_by_metadata = None

        return filter_by_metadata
