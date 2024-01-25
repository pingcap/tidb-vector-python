import numpy as np

from sqlalchemy.types import Float, UserDefinedType


class VectorType(UserDefinedType):
    """
    Represents a user-defined type for storing vector in TiDB
    """

    cache_ok = True

    def __init__(self, dim=None):
        super(UserDefinedType, self).__init__()
        self.dim = None

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
        return "VECTOR(%d)" % self.dim

    def bind_processor(self, dialect):
        """Convert the vector float array to a string representation suitable for binding to a database column."""

        def process(value, dim=None):
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

        return process

    def result_processor(self, dialect, coltype):
        """Convert the vector data from the database into vector array."""

        def process(value):
            if value is None or isinstance(value, np.ndarray):
                return value

            return np.array(value[1:-1].split(","), dtype=np.float32)

        return process

    class comparator_factory(UserDefinedType.Comparator):
        """Returns a comparator factory that provides the distance functions."""

        def l2_distance(self, other):
            return self.op("<-->", return_type=Float)(other)

        def cosine_distance(self, other):
            return self.op("<==>", return_type=Float)(other)
