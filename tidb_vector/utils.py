import typing

import numpy as np


def encode_vector(value: typing.Union[np.ndarray, typing.List[float]], dim=None):
    if value is None:
        return value

    if dim is not None and len(value) != dim:
        raise ValueError(f"expected {dim} dimensions, but got {len(value)}")

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("expected ndim to be 1")
        return f"[{','.join(map(str, value))}]"

    return str(value)


def decode_vector(value: str) -> np.ndarray:
    if value is None:
        return value

    if value == "[]":
        return np.array([], dtype=np.float32)

    return np.array(value[1:-1].split(","), dtype=np.float32)
