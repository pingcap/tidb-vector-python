# tidb-vector-python

This is a Python client for TiDB Vector.

> Now only TiDB Cloud Serverless cluster support vector data type, see this [blog](https://www.pingcap.com/blog/integrating-vector-search-into-tidb-for-ai-applications/) for more information.

## Installation

```bash
pip install tidb-vector
```

## Usage

TiDB vector supports below distance functions:

- `L1Distance`
- `L2Distance`
- `CosineDistance`
- `NegativeInnerProduct`

supports following orm or framework:

- [SQLAlchemy](#sqlalchemy)
- [Django](#django)

### SQLAlchemy

#### Define table with vector field

```python
from sqlalchemy import Column, Integer
from sqlalchemy.orm import declarative_base
from tidb_vector.sqlalchemy import VectorType

Base = declarative_base()

class Test(Base):
    __tablename__ = 'test'
    id = Column(Integer, primary_key=True)
    embedding = Column(VectorType(3))
```

#### Insert vector data

```python
test = Test(embedding=[1, 2, 3])
session.add(test)
session.commit()
```

#### Query with vector data

Get the nearest neighbors

```python
session.scalars(select(Test).order_by(Test.embedding.l2_distance([3, 1, 2])).limit(5))
```

Get the distance

```python
session.scalars(select(Test.embedding.l2_distance([3, 1, 2])))
```

Get within a certain distance

```python
session.scalars(select(Test).filter(Test.embedding.l2_distance([3, 1, 2]) < 5))
```

### Django

To use vector field in Django, you need to use [`django-tidb`](https://github.com/pingcap/django-tidb?tab=readme-ov-file#vectorfield-beta).
