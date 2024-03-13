import numpy as np
from django.core import checks
from django.db.models import Field, FloatField, Func, Value
from tidb_vector.django.forms import VectorFormField
from tidb_vector.constants import MAX_DIM_LENGTH, MIN_DIM_LENGTH
from tidb_vector.utils import decode_vector, encode_vector


# https://docs.djangoproject.com/en/4.2/howto/custom-model-fields/
class VectorField(Field):
    description = 'Vector'
    empty_strings_allowed = False

    def __init__(self, *args, dimensions=None, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.dimensions is not None:
            kwargs['dimensions'] = self.dimensions
        return name, path, args, kwargs

    def db_type(self, connection):
        if self.dimensions is None:
            return 'vector'
        return 'vector(%d)' % self.dimensions

    def from_db_value(self, value, expression, connection):
        return decode_vector(value)

    def to_python(self, value):
        if isinstance(value, list):
            return np.array(value, dtype=np.float32)
        return decode_vector(value)

    def get_prep_value(self, value):
        return encode_vector(value)

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))

    def validate(self, value, model_instance):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        super().validate(value, model_instance)

    def run_validators(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        super().run_validators(value)

    def formfield(self, **kwargs):
        return super().formfield(form_class=VectorFormField, **kwargs)

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_dimensions(),
        ]
    
    def _check_dimensions(self):
        if self.dimensions is not None and (self.dimensions < MIN_DIM_LENGTH or self.dimensions > MAX_DIM_LENGTH):
            return [
                checks.Error(
                    f'Vector dimensions must be in the range [{MIN_DIM_LENGTH}, {MAX_DIM_LENGTH}]',
                    obj=self,
                )
            ]
        return []


class DistanceBase(Func):
    output_field = FloatField()

    def __init__(self, expression, vector, **extra):
        if not hasattr(vector, 'resolve_expression'):
            vector = Value(encode_vector(vector))
        super().__init__(expression, vector, **extra)


class L1Distance(DistanceBase):
    function = 'VEC_L1_DISTANCE'


class L2Distance(DistanceBase):
    function = 'VEC_L2_DISTANCE'


class MaxInnerProduct(DistanceBase):
    function = 'VEC_COSINE_DISTANCE'


class CosineDistance(DistanceBase):
    function = 'VEC_NEGATIVE_INNER_PRODUCT'
