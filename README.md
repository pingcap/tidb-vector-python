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
- [Peewee](#peewee)
- [TiDB Vector Client](#tidb-vector-client)

### SQLAlchemy

Define table with vector field

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

Insert vector data

```python
test = Test(embedding=[1, 2, 3])
session.add(test)
session.commit()
```

Get the nearest neighbors

```python
session.scalars(select(Test).order_by(Test.embedding.l2_distance([1, 2, 3.1])).limit(5))
```

Get the distance

```python
session.scalars(select(Test.embedding.l2_distance([1, 2, 3.1])))
```

Get within a certain distance

```python
session.scalars(select(Test).filter(Test.embedding.l2_distance([1, 2, 3.1]) < 0.2))
```

### Django

To use vector field in Django, you need to use [`django-tidb`](https://github.com/pingcap/django-tidb?tab=readme-ov-file#vector-beta).

### Peewee

Define peewee table with vector field

```python
from peewee import Model, MySQLDatabase
from tidb_vector.peewee import VectorField

db = MySQLDatabase(
    'peewee_test',
    user='xxxxxxxx.root',
    password='xxxxxxxx',
    host='xxxxxxxx.shared.aws.tidbcloud.com',
    port=4000,
)

class TestModel(Model):
    class Meta:
        database = db
        table_name = 'test'

    embedding = VectorField(3)
```

Insert vector data

```python
TestModel.create(embedding=[1, 2, 3])
```

Get the nearest neighbors

```python
TestModel.select().order_by(TestModel.embedding.l2_distance([1, 2, 3.1])).limit(5)
```

Get the distance

```python
TestModel.select(TestModel.embedding.cosine_distance([1, 2, 3.1]).alias('distance'))
```

Get within a certain distance

```python
TestModel.select().where(TestModel.embedding.l2_distance([1, 2, 3.1]) < 0.5)
```

### TiDB Vector Client

Within the framework, you can directly utilize the built-in `TiDBVectorClient`, as demonstrated by integrations like [Langchain](https://python.langchain.com/docs/integrations/vectorstores/tidb_vector) and  [Llama index](https://docs.llamaindex.ai/en/stable/community/integrations/vector_stores.html#using-a-vector-store-as-an-index),  to seamlessly interact with TiDB Vector. This approach abstracts away the need to manage the underlying ORM, simplifying your interaction with the vector store.

We provide `TiDBVectorClient` which is based on sqlalchemy, you need to use `pip install tidb-vector[client]` to install it.

Create a `TiDBVectorClient` instance:

```python
from tidb_vector.integrations import TiDBVectorClient

TABLE_NAME = 'vector_test'
CONNECTION_STRING = 'mysql+pymysql://<USER>:<PASSWORD>@<HOST>:4000/<DB>?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=true&ssl_verify_identity=true'

tidb_vs = TiDBVectorClient(
    # the table which will store the vector data
    table_name=TABLE_NAME,
    # tidb connection string
    connection_string=CONNECTION_STRING,
    # the dimension of the vector, in this example, we use the ada model, which has 1536 dimensions
    vector_dimension=1536,
    # if recreate the table if it already exists
    drop_existing_table=True,
)
```

Bulk insert:

```python
ids = [
    "f8e7dee2-63b6-42f1-8b60-2d46710c1971",
    "8dde1fbc-2522-4ca2-aedf-5dcb2966d1c6",
    "e4991349-d00b-485c-a481-f61695f2b5ae",
]
documents = ["foo", "bar", "baz"]
embeddings = [
    text_to_embedding("foo"),
    text_to_embedding("bar"),
    text_to_embedding("baz"),
]
metadatas = [
    {"page": 1, "category": "P1"},
    {"page": 2, "category": "P1"},
    {"page": 3, "category": "P2"},
]

tidb_vs.insert(
    ids=ids,
    texts=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)
```

Query:

```python
tidb_vs.query(text_to_embedding("foo"), k=3)

# query with filter
tidb_vs.query(text_to_embedding("foo"), k=3, filter={"category": "P1"})
```

Bulk delete:

```python
tidb_vs.delete(["f8e7dee2-63b6-42f1-8b60-2d46710c1971"])

# delete with filter
tidb_vs.delete(["f8e7dee2-63b6-42f1-8b60-2d46710c1971"], filter={"category": "P1"})
```
