import numpy as np


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

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    return np.array(value[1:-1].split(","), dtype=np.float32)
