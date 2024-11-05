from django.db import models
from django_tidb.fields.vector import VectorField, VectorIndex, CosineDistance, L2Distance


class Document(models.Model):
    content = models.TextField()
    embedding = VectorField(dimensions=3)
    class Meta:
        indexes = [
            VectorIndex(L2Distance("embedding"), name='idx_l2'),
            VectorIndex(CosineDistance("embedding"), name='idx_cos'),
        ]
