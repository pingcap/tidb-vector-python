import enum

# TiDB Vector has a limitation on the dimension length
MAX_DIM = 16000
MIN_DIM = 1


class DistanceMetric(enum.Enum):
    L2 = "L2"
    COSINE = "COSINE"

    def to_sql_func(self):
        if self == DistanceMetric.L2:
            return "VEC_L2_DISTANCE"
        elif self == DistanceMetric.COSINE:
            return "VEC_COSINE_DISTANCE"
        else:
            raise ValueError("unsupported distance metric")
