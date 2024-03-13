# tidb-vector-python

This is a python client for TiDB Vector, currently TiDB only supports vector data type in TiDB Cloud Serverless.

## Install

```bash
pip install tidb_vector
```

## Usage

### Django

For Django, you need to install `django-tidb` at first, see its installation guide at [django-tidb](https://github.com/pingcap/django-tidb?tab=readme-ov-file#installation-guide).

#### Define a model with vector field

```python
from django.db import models
from tidb_vector.django import VectorField

class Test(models.Model):
    embedding = VectorField(dimensions=3)
```

#### Create a record

```python
Test.objects.create(embedding=[1, 2, 3])
```

#### Get instances with vector field

TiDB Vector support below distance functions:

- `L1Distance`
- `L2Distance`
- `CosineDistance`
- `NegativeInnerProduct`

Get instances with vector field and calculate distance to a given vector:

```python
Test.objects.annotate(distance=CosineDistance('embedding', [3, 1, 2]))
```

Get instances with vector field and calculate distance to a given vector, and filter by distance:

```python
Test.objects.alias(distance=CosineDistance('embedding', [3, 1, 2])).filter(distance__lt=5)
```

### SQLAlchemy

TODO
