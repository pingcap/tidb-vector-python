import typing
import sqlalchemy
import tidb_vector
import tidb_vector.utils


class VectorType(sqlalchemy.types.UserDefinedType):
    """
    Represents a vector column type in TiDB.
    """

    dim: typing.Optional[int]

    cache_ok = True

    def __init__(self, dim=None):
        if dim is not None and not isinstance(dim, int):
            raise ValueError("expected dimension to be an integer or None")

        # tidb vector dimention length has limitation
        if dim is not None and (dim < tidb_vector.MIN_DIM or dim > tidb_vector.MAX_DIM):
            raise ValueError(
                f"expected dimension to be in [{tidb_vector.MIN_DIM}, {tidb_vector.MAX_DIM}]"
            )

        super(sqlalchemy.types.UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw):
        """
        Returns the column specification for the vector column.

        If the dimension is not specified, it returns "VECTOR".
        Otherwise, it returns "VECTOR(<dimension>)".

        :param kw: Additional keyword arguments.
        :return: The column specification string.
        """

        if self.dim is None:
            return "VECTOR"
        return f"VECTOR({self.dim})"

    def bind_processor(self, dialect):
        """Convert the vector float array to a string representation suitable for binding to a database column."""

        def process(value):
            return tidb_vector.utils.encode_vector(value, self.dim)

        return process

    def result_processor(self, dialect, coltype):
        """Convert the vector data from the database into vector array."""

        def process(value):
            return tidb_vector.utils.decode_vector(value)

        return process

    class comparator_factory(sqlalchemy.types.UserDefinedType.Comparator):
        """Returns a comparator factory that provides the distance functions."""

        def l1_distance(self, other):
            formatted_other = tidb_vector.utils.encode_vector(other)
            return sqlalchemy.func.VEC_L1_DISTANCE(self, formatted_other).label(
                "l1_distance"
            )

        def l2_distance(self, other):
            formatted_other = tidb_vector.utils.encode_vector(other)
            return sqlalchemy.func.VEC_L2_DISTANCE(self, formatted_other).label(
                "l2_distance"
            )

        def cosine_distance(self, other):
            formatted_other = tidb_vector.utils.encode_vector(other)
            return sqlalchemy.func.VEC_COSINE_DISTANCE(self, formatted_other).label(
                "cosine_distance"
            )

        def negative_inner_product(self, other):
            formatted_other = tidb_vector.utils.encode_vector(other)
            return sqlalchemy.func.VEC_NEGATIVE_INNER_PRODUCT(
                self, formatted_other
            ).label("negative_inner_product")
