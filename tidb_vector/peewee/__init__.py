from peewee import Field, fn

from tidb_vector.utils import decode_vector, encode_vector


class VectorField(Field):
    field_type = "VECTOR"

    def __init__(self, dimensions=None, *args, **kwargs):
        self.dimensions = dimensions
        super(VectorField, self).__init__(*args, **kwargs)

    def get_modifiers(self):
        return self.dimensions and [self.dimensions] or None

    def db_value(self, value):
        return encode_vector(value)

    def python_value(self, value):
        return decode_vector(value)

    def l1_distance(self, vector):
        return fn.VEC_L1_DISTANCE(self, self.to_value(vector))

    def l2_distance(self, vector):
        return fn.VEC_L2_DISTANCE(self, self.to_value(vector))

    def cosine_distance(self, vector):
        return fn.VEC_COSINE_DISTANCE(self, self.to_value(vector))

    def negative_inner_product(self, vector):
        return fn.VEC_NEGATIVE_INNER_PRODUCT(self, self.to_value(vector))
