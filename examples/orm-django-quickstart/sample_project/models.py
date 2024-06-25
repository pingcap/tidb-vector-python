from django.db import models
from django_tidb.fields.vector import VectorField


class Document(models.Model):
    content = models.TextField()
    embedding = VectorField(dimensions=3)


class DocumentWithIndex(models.Model):
    content = models.TextField()
    # Note:
    #   - Using comment to add hnsw index is a temporary solution. In the future it will use `CREATE INDEX` syntax.
    #   - Currently the HNSW index cannot be changed after the table has been created.
    #   - Only Django >= 4.2 supports `db_comment`.
    embedding = VectorField(dimensions=3, db_comment="hnsw(distance=cosine)")
