from peewee import Expression, Field

from tidb_vector.utils import decode_vector, encode_vector


class VectorField(Field):
    field_type = 'VECTOR<FLOAT>'

    def __init__(self, dimensions=None, *args, **kwargs):
        self.dimensions = dimensions
        super(VectorField, self).__init__(*args, **kwargs)

    def get_modifiers(self):
        return self.dimensions and [self.dimensions] or None

    def db_value(self, value):
        return encode_vector(value)

    def python_value(self, value):
        return decode_vector(value)

    def _distance(self, op, vector):
        return Expression(lhs=self, op=op, rhs=self.to_value(vector))

    def l1_distance(self, vector):
        return self._distance("VEC_L1_DISTANCE", vector)

    def l2_distance(self, vector):
        return self._distance("VEC_L2_DISTANCE", vector)

    def cosine_distance(self, vector):
        return self._distance('VEC_COSINE_DISTANCE', vector)

    def negative_inner_product(self, vector):
        return self._distance("VEC_NEGATIVE_INNER_PRODUCT", vector)
