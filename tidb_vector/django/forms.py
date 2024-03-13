import numpy as np
from django import forms
from tidb_vector.django.widgets import VectorWidget


class VectorFormField(forms.CharField):
    widget = VectorWidget

    def has_changed(self, initial, data):
        if isinstance(initial, np.ndarray):
            initial = initial.tolist()
        return super().has_changed(initial, data)
