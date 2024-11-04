import sqlalchemy
import tidb_vector
import tidb_vector.sqlalchemy


class VectorAdaptor:
    """
    A wrapper over existing SQLAlchemy engine to provide additional vector search capabilities.
    """

    engine: sqlalchemy.Engine

    def __init__(self, engine: sqlalchemy.Engine):
        self.engine = engine

    def _check_vector_column(self, column: sqlalchemy.Column):
        if not isinstance(column.type, tidb_vector.sqlalchemy.VectorType):
            raise ValueError("Not a vector column")

    def has_vector_index(self, column: sqlalchemy.Column) -> bool:
        """
        Check if the index for the vector column exists.
        """

        self._check_vector_column(column)

        with self.engine.begin() as conn:
            table_name = conn.dialect.identifier_preparer.format_table(column.table)
            query = sqlalchemy.text(f"SHOW INDEX FROM {table_name}")
            result = conn.execute(query)
            result_dict = result.mappings().all()
            for row in result_dict:
                if row["Column_name"].lower() == column.name.lower():
                    return True
        return False

    def create_vector_index(
        self,
        column: sqlalchemy.Column,
        distance_metric: tidb_vector.DistanceMetric,
        skip_existing: bool = False,
    ):
        """
        Create the index for the vector column.
        """

        self._check_vector_column(column)

        if column.type.dim is None:
            raise ValueError(
                "Vector index is only supported for fixed dimension vectors"
            )

        if skip_existing:
            if self.has_vector_index(column):
                # TODO: Currently there is no easy way to verify whether the distance
                # metric is correct. We should check it and throw error if distance metric is not matching
                return

        with self.engine.begin() as conn:
            table_name = conn.dialect.identifier_preparer.format_table(column.table)
            column_name = conn.dialect.identifier_preparer.format_column(column)
            index_name = conn.dialect.identifier_preparer.quote(
                f"vec_idx_{column.name}"
            )

            query = sqlalchemy.text(f"ALTER TABLE {table_name} SET TIFLASH REPLICA 1")
            conn.execute(query)

            query = sqlalchemy.text(
                f"""
                ALTER TABLE {table_name}
                ADD VECTOR INDEX {index_name} (({distance_metric.to_sql_func()}({column_name})))
                """
            )
            conn.execute(query)
