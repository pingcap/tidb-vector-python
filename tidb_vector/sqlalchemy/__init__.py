from sqlalchemy.types import UserDefinedType
from sqlalchemy.sql import func

from tidb_vector.constants import MAX_DIMENSION_LENGTH, MIN_DIMENSION_LENGTH
from tidb_vector.utils import decode_vector, encode_vector


class VectorType(UserDefinedType):
    """
    Represents a user-defined type for storing vector in TiDB
    """

    cache_ok = True

    def __init__(self, dim=None):
        if dim is not None and not isinstance(dim, int):
            raise ValueError("expected dimension to be an integer or None")

        # tidb vector dimention length has limitation
        if dim is not None and (
            dim < MIN_DIMENSION_LENGTH or dim > MAX_DIMENSION_LENGTH
        ):
            raise ValueError(
                f"expected dimension to be in [{MIN_DIMENSION_LENGTH}, {MAX_DIMENSION_LENGTH}]"
            )

        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw):
        """
        Returns the column specification for the vector column.

        If the dimension is not specified, it returns "VECTOR<FLOAT>".
        Otherwise, it returns "VECTOR(<dimension>)".

        :param kw: Additional keyword arguments.
        :return: The column specification string.
        """

        if self.dim is None:
            return "VECTOR<FLOAT>"
        return "VECTOR<FLOAT>(%d)" % self.dim

    def bind_processor(self, dialect):
        """Convert the vector float array to a string representation suitable for binding to a database column."""

        def process(value):
            return encode_vector(value, self.dim)

        return process

    def result_processor(self, dialect, coltype):
        """Convert the vector data from the database into vector array."""

        def process(value):
            return decode_vector(value)

        return process

    class comparator_factory(UserDefinedType.Comparator):
        """Returns a comparator factory that provides the distance functions."""

        def l1_distance(self, other):
            formatted_other = encode_vector(other)
            return func.VEC_L1_DISTANCE(self, formatted_other).label("l1_distance")

        def l2_distance(self, other):
            formatted_other = encode_vector(other)
            return func.VEC_L2_DISTANCE(self, formatted_other).label("l2_distance")

        def cosine_distance(self, other):
            formatted_other = encode_vector(other)
            return func.VEC_COSINE_DISTANCE(self, formatted_other).label(
                "cosine_distance"
            )

        def negative_inner_product(self, other):
            formatted_other = encode_vector(other)
            return func.VEC_NEGATIVE_INNER_PRODUCT(self, formatted_other).label(
                "negative_inner_product"
            )
