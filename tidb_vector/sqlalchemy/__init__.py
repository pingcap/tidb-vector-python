import numpy as np

from sqlalchemy.types import UserDefinedType
from sqlalchemy.sql import func


def encode_vector(value, dim=None):
    if value is None:
        return value

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("expected ndim to be 1")

        if not np.issubdtype(value.dtype, np.integer) and not np.issubdtype(
            value.dtype, np.floating
        ):
            raise ValueError("dtype must be numeric")

        value = value.tolist()

    if dim is not None and len(value) != dim:
        raise ValueError("expected %d dimensions, not %d" % (dim, len(value)))

    return "[" + ",".join([str(float(v)) for v in value]) + "]"


def decode_vector(value):
    if value is None or isinstance(value, np.ndarray):
        return value

    return np.array(value[1:-1].split(","), dtype=np.float32)


class VectorType(UserDefinedType):
    """
    Represents a user-defined type for storing vector in TiDB
    """

    cache_ok = True

    def __init__(self, dim=None):
        if dim is not None and not isinstance(dim, int):
            raise ValueError("expected dimension to be an integer or None")

        # tidb vector dim length is allowed to be in [1,16000]
        if dim is not None and (dim < 1 or dim > 16000):
            raise ValueError("expected dimension to be in [1,16000]")

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
