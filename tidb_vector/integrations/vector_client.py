import contextlib
import copy
from dataclasses import dataclass
import logging
import enum
import uuid
from typing import Type, Tuple, Any, Dict, Generator, Iterable, List, Optional

import sqlalchemy
from sqlalchemy.orm import Session, declarative_base
from tidb_vector.sqlalchemy import VectorType
from tidb_vector.integrations.utils import (
    get_embedding_column_definition,
    EmbeddingColumnMismatchError,
)

logger = logging.getLogger()


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    # INNER_PRODUCT = "inner_product"


def _create_vector_table_model(
    table_name: str,
    dim: Optional[int] = None,
    distance: Optional[DistanceStrategy] = None,
) -> Tuple[Type[declarative_base], Type]:
    """Create a vector model class."""

    OrmBase = declarative_base()  # type: Any

    class VectorTableModel(OrmBase):
        """
        embedding: The column to store the vector data.
        document: The column to store the document content.
        meta: The column to store the metadata of the document.
            It can be used to filter the document when performing search
            e.g. {"title": "The title of the document", "custom_id": "123"}
        """

        __tablename__ = table_name
        id = sqlalchemy.Column(
            sqlalchemy.String(36), primary_key=True, default=lambda: str(uuid.uuid4())
        )
        embedding = sqlalchemy.Column(
            VectorType(dim),  # Using the VectorType to store the vector data
            nullable=False,  # Assuming non-nullability as before
            comment="" if distance is None else f"hnsw(distance={distance})",
        )
        document = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
        meta = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)
        create_time = sqlalchemy.Column(
            sqlalchemy.DateTime, server_default=sqlalchemy.text("CURRENT_TIMESTAMP")
        )
        update_time = sqlalchemy.Column(
            sqlalchemy.DateTime,
            server_default=sqlalchemy.text(
                "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            ),
        )

    return OrmBase, VectorTableModel


@dataclass
class QueryResult:
    id: str
    document: str
    metadata: dict
    distance: float


class TiDBVectorClient:
    def __init__(
        self,
        connection_string: str,
        table_name: str,
        distance_strategy: Optional[DistanceStrategy] = None,
        vector_dimension: Optional[int] = None,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        drop_existing_table: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a vector client in a specified table within a TiDB database.

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
        self._vector_dimension = vector_dimension
        self._table_name = table_name
        self._engine_args = engine_args or {}
        self._drop_existing_table = drop_existing_table
        self._bind = self._create_engine()
        self._check_table_compatibility()  # check if the embedding is compatible
        self._orm_base, self._table_model = _create_vector_table_model(
            table_name, vector_dimension, distance_strategy
        )
        _ = self.distance_strategy  # check if distance strategy is valid
        self._create_table_if_not_exists()

    def _check_table_compatibility(self) -> None:
        """
        Check if the table is compatible with the current configuration.
        """
        if self._drop_existing_table:
            return

        actual_dim, actual_distance_strategy = get_embedding_column_definition(
            connection_string=self.connection_string,
            table_name=self._table_name,
            column_name="embedding",
            engine_args=self._engine_args,
        )
        if actual_dim is not None:
            # If the vector dimension is not set, set it to the actual dimension
            if self._vector_dimension is None:
                self._vector_dimension = actual_dim
            elif actual_dim != self._vector_dimension:
                raise EmbeddingColumnMismatchError(
                    existing_col=f"vector({actual_dim})",
                    expected_col=f"vector({self._vector_dimension})",
                )

        if actual_distance_strategy is not None:
            if self._distance_strategy is None:
                self._distance_strategy = DistanceStrategy(actual_distance_strategy)
            elif actual_distance_strategy != self._distance_strategy:
                raise EmbeddingColumnMismatchError(
                    existing_col=f"vector({actual_dim}) COMMENT 'hnsw(distance={actual_distance_strategy})'",
                    expected_col=f"vector({self._vector_dimension}) COMMENT 'hnsw(distance={self._distance_strategy})'",
                )

    def _create_table_if_not_exists(self) -> None:
        """
        If the `self._pre_delete_table` flag is set,
        the existing table will be dropped before creating a new one.
        """
        if self._drop_existing_table:
            self.drop_table()
        with Session(self._bind) as session, session.begin():
            self._orm_base.metadata.create_all(session.get_bind())
            # wait for tidb support vector index

    def drop_table(self) -> None:
        """Drops the table if it exists."""
        with Session(self._bind) as session, session.begin():
            self._orm_base.metadata.drop_all(session.get_bind())

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        """Create a sqlalchemy engine."""
        return sqlalchemy.create_engine(url=self.connection_string, **self._engine_args)

    def __deepcopy__(self, memo):
        # Create a shallow copy of the object to start with, to copy non-engine attributes
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except the engine connection (_bind)
        for k, v in self.__dict__.items():
            if k != "_bind":  # Skip copying the engine connection
                setattr(result, k, copy.deepcopy(v, memo))

        # Directly assign the engine connection without copying
        result._bind = self._bind

        return result

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
            return self._table_model.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self._table_model.embedding.cosine_distance
        # elif self._distance_strategy == DistanceStrategy.INNER_PRODUCT:
        #    return self._table_model.embedding.negative_inner_product
        elif self._distance_strategy is None:  # default to cosine
            return self._table_model.embedding.cosine_distance
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

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
                embeded_doc = self._table_model(
                    id=id,
                    embedding=embedding,
                    document=text,
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
                filter_by = sqlalchemy.and_(self._table_model.id.in_(ids), filter_by)
            stmt = sqlalchemy.delete(self._table_model).filter(filter_by)
            session.execute(stmt)
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
            post_filter_enabled (bool, optional): Whether to apply the post-filtering.
                TiDB cannot utilize Vector Index when query contains a pre-filter.
            post_filter_multiplier (int, optional): A multiplier to increase the initial
                number of results fetched before applying the filter. Defaults to 1.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of tuples containing relevant documents and their similarity scores.
        """
        relevant_docs = self._vector_search(query_vector, k, filter, **kwargs)

        return [
            QueryResult(
                document=doc.document,
                metadata=doc.meta,
                id=doc.id,
                distance=doc.distance,
            )
            for doc in relevant_docs
        ]

    def _vector_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """vector search from table."""

        post_filter_enabled = kwargs.get("post_filter_enabled", False)
        post_filter_multiplier = kwargs.get("post_filter_multiplier", 1)
        with Session(self._bind) as session:
            if post_filter_enabled is False or not filter:
                filter_by = self._build_filter_clause(filter)
                results = (
                    session.query(
                        self._table_model.id,
                        self._table_model.meta,
                        self._table_model.document,
                        self.distance_strategy(query_embedding).label("distance"),
                    )
                    .filter(filter_by)
                    .order_by(sqlalchemy.asc("distance"))
                    .limit(k)
                    .all()
                )
            else:
                # Caused by the tidb vector search plan limited, this post_filter_multiplier is used to
                # improved the search performance temporarily.
                # Notice the return count may be less than k in this situation.
                subquery = (
                    session.query(
                        self._table_model.id,
                        self._table_model.meta,
                        self._table_model.document,
                        self.distance_strategy(query_embedding).label("distance"),
                    )
                    .order_by(sqlalchemy.asc("distance"))
                    .limit(post_filter_multiplier * k * 10)
                    .subquery()
                )
                filter_by = self._build_filter_clause(filter, subquery.c)
                results = (
                    session.query(
                        subquery.c.id,
                        subquery.c.meta,
                        subquery.c.document,
                        subquery.c.distance,
                    )
                    .filter(filter_by)
                    .order_by(sqlalchemy.asc(subquery.c.distance))
                    .limit(k)
                    .all()
                )
        return results

    def _build_filter_clause(
        self,
        filters: Optional[Dict[str, Any]] = None,
        table_model: Optional[Any] = None,
    ) -> Any:
        """
        Builds the filter clause for querying based on the provided filters.

        Args:
            filter (Dict[str, str]): The filter conditions to apply.

        Returns:
            Any: The filter clause to be used in the query on TiDB.
        """

        if table_model is None:
            table_model = self._table_model

        filter_by = sqlalchemy.true()
        if filters is not None:
            filter_clauses = []

            for key, value in filters.items():
                if key.lower() == "$and":
                    and_clauses = [
                        self._build_filter_clause(condition, table_model)
                        for condition in value
                        if isinstance(condition, dict) and condition is not None
                    ]
                    filter_by_metadata = sqlalchemy.and_(*and_clauses)
                    filter_clauses.append(filter_by_metadata)
                elif key.lower() == "$or":
                    or_clauses = [
                        self._build_filter_clause(condition, table_model)
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
                    filter_by_metadata = self._create_filter_clause(
                        table_model, key, value
                    )

                    if filter_by_metadata is not None:
                        filter_clauses.append(filter_by_metadata)
                else:
                    filter_by_metadata = (
                        sqlalchemy.func.json_extract(table_model.meta, f"$.{key}")
                        == value
                    )
                    filter_clauses.append(filter_by_metadata)

            filter_by = sqlalchemy.and_(filter_by, *filter_clauses)
        return filter_by

    def _create_filter_clause(self, table_model, key, value):
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

        json_key = sqlalchemy.func.json_extract(table_model.meta, f"$.{key}")
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

    def execute(self, sql: str, params: Optional[dict] = None) -> dict:
        """
        Execute an arbitrary SQL command and return execution status and result.

        This method can handle both DML (Data Manipulation Language) commands such as INSERT, UPDATE, DELETE,
        and DQL (Data Query Language) commands like SELECT. It returns a structured dictionary indicating
        the execution success status, result (for SELECT queries or affected rows count for DML), and any
        error message if the execution failed.

        Args:
            sql (str): The SQL command to execute.
            params (Optional[dict]): Parameters to bind to the SQL command, if any.

        Returns:
            dict: A dictionary containing 'success': boolean indicating if the execution was successful,
                'result': fetched results for SELECT or affected rows count for other statements,
                and 'error': error message if execution failed.

        Examples:
            - Creating a table:
            execute("CREATE TABLE users (id INT, username VARCHAR(50), email VARCHAR(50))")
            This would return: {'success': True, 'result': 0, 'error': None}

            - Executing a SELECT query:
            execute("SELECT * FROM users WHERE username = :username", {"username": "john_doe"})
            This would return: {'success': True, 'result': [(user data)], 'error': None}

            - Inserting data into a table:
            execute(
                "INSERT INTO users (username, email) VALUES (:username, :email)",
                {"username": "new_user", "email": "new_user@example.com"}
            )
            This would return: {'success': True, 'result': 1, 'error': None} if one row was affected.

            - Handling an error (e.g., table does not exist):
            execute("SELECT * FROM non_existing_table")
            This might return: {'success': False, 'result': None, 'error': '(Error message)'}
        """
        try:
            with Session(self._bind) as session, session.begin():
                result = session.execute(sqlalchemy.text(sql), params)
                session.commit()  # Ensure changes are committed for non-SELECT statements.
                if sql.strip().lower().startswith("select"):
                    return {"success": True, "result": result.fetchall(), "error": None}
                else:
                    return {"success": True, "result": result.rowcount, "error": None}
        except Exception as e:
            # Log the error or handle it as needed
            logger.error(f"SQL execution error: {str(e)}")
            return {"success": False, "result": None, "error": str(e)}
