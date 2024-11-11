import peewee
import tidb_vector
from .vector_type import VectorField


class VectorAdaptor:
    """
    A wrapper over existing Peewee Database to provide additional vector search capabilities.
    """

    engine: peewee.Database

    def __init__(self, engine: peewee.Database):
        self.engine = engine

    def _check_vector_column(self, field: VectorField):
        if not isinstance(field, VectorField):
            raise ValueError("Not a vector field")

    def has_vector_index(self, field: VectorField) -> bool:
        """
        Check if the index for the vector column exists.
        """

        self._check_vector_column(field)

        table_name = field.model._meta.table_name

        # TODO: Better quote
        cursor: peewee.CursorWrapper = self.engine.execute_sql(
            f"SHOW INDEX FROM `{table_name}`"
        )
        column_name_idx = None
        for idx, column in enumerate(cursor.description):
            if column[0].lower() == "column_name":
                column_name_idx = idx
                break
        if column_name_idx is None:
            raise ValueError("Failed to parse SHOW INDEX result")

        for row in cursor:
            column_name = row[column_name_idx]
            if column_name.lower() == field.name.lower():
                return True

        return False

    def create_vector_index(
        self,
        field: VectorField,
        distance_metric: tidb_vector.DistanceMetric,
        skip_existing: bool = False,
    ):
        """
        Create vector index for the vector column.

        Parameters
        ----------
        field : peewee.Field
            The field for which the vector index is to be created.

        distance_metric : tidb_vector.DistanceMetric
            The distance metric to be used for the vector index.
                Available values are:
                - tidb_vector.DistanceMetric.L2
                - tidb_vector.DistanceMetric.COSINE

        skip_existing : bool
            If True, skips creating the index if it already exists. Default is False.

        Raises
        ------
        ValueError
            If the vector field does not have a fixed dimension.

        ValueError
            If the field is not a vector field.

        Note
        ----
        If you want to use high-avaliability columnar storage feature, use raw SQL instead.

        """

        self._check_vector_column(field)

        if field.dimensions is None:
            raise ValueError(
                "Vector index is only supported for fixed dimension vectors"
            )

        if skip_existing:
            if self.has_vector_index(field):
                # TODO: Currently there is no easy way to verify whether the distance
                # metric is correct. We should check it and throw error if distance metric is not matching
                return

        table_name = field.model._meta.table_name
        column_name = field.name
        index_name = f"vec_idx_{field.name}"

        self.engine.execute_sql(f"ALTER TABLE `{table_name}` SET TIFLASH REPLICA 1")
        self.engine.execute_sql(
            f"""
            ALTER TABLE `{table_name}`
            ADD VECTOR INDEX `{index_name}` (({distance_metric.to_sql_func()}(`{column_name}`)))
            """
        )
