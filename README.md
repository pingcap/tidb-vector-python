# tidb-vector-python

This is a Python client for TiDB Vector.

> Now only TiDB Cloud Serverless cluster support vector data type, see this [docs](https://docs.pingcap.com/tidbcloud/vector-search-overview?utm_source=github&utm_medium=tidb-vector-python) for more information.

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

It also supports using hnsw index with l2 or cosine distance to speed up the search, for more details see [Vector Search Indexes in TiDB](https://docs.pingcap.com/tidbcloud/vector-search-index)

Supports following orm or framework:

- [SQLAlchemy](#sqlalchemy)
- [Django](#django)
- [Peewee](#peewee)
- [TiDB Vector Client](#tidb-vector-client)

### SQLAlchemy

Learn how to connect to TiDB Serverless in the [TiDB Cloud documentation](https://docs.pingcap.com/tidbcloud/dev-guide-sample-application-python-sqlalchemy).

Define table with vector field

```python
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.orm import declarative_base
from tidb_vector.sqlalchemy import VectorType

engine = create_engine('mysql://****.root:******@gateway01.xxxxxx.shared.aws.tidbcloud.com:4000/test')
Base = declarative_base()

class Test(Base):
    __tablename__ = 'test'
    id = Column(Integer, primary_key=True)
    embedding = Column(VectorType(3))

# or add hnsw index when creating table
class TestWithIndex(Base):
    __tablename__ = 'test_with_index'
    id = Column(Integer, primary_key=True)
    embedding = Column(VectorType(3), comment="hnsw(distance=l2)")

Base.metadata.create_all(engine)
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

# Using `pymysql` as the driver
connect_kwargs = {
    'ssl_verify_cert': True,
    'ssl_verify_identity': True,
}

# Using `mysqlclient` as the driver
connect_kwargs = {
    'ssl_mode': 'VERIFY_IDENTITY',
    'ssl': {
        # Root certificate default path
        # https://docs.pingcap.com/tidbcloud/secure-connections-to-serverless-clusters/#root-certificate-default-path
        'ca': '/etc/ssl/cert.pem'  # MacOS
    },
}

db = MySQLDatabase(
    'peewee_test',
    user='xxxxxxxx.root',
    password='xxxxxxxx',
    host='xxxxxxxx.shared.aws.tidbcloud.com',
    port=4000,
    **connect_kwargs,
)

class TestModel(Model):
    class Meta:
        database = db
        table_name = 'test'

    embedding = VectorField(3)

# or add hnsw index when creating table
class TestModelWithIndex(Model):
    class Meta:
        database = db
        table_name = 'test_with_index'

    embedding = VectorField(3, constraints=[SQL("COMMENT 'hnsw(distance=l2)'")])


db.connect()
db.create_tables([TestModel, TestModelWithIndex])
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
CONNECTION_STRING = 'mysql+pymysql://<USER>:<PASSWORD>@<HOST>:4000/<DB>?ssl_verify_cert=true&ssl_verify_identity=true'

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

## Examples

There are some examples to show how to use the tidb-vector-python to interact with TiDB Vector in different scenarios.

- [OpenAI Embedding](./examples/openai_embedding/README.md): use the OpenAI embedding model to generate vectors for text data, store them in TiDB Vector, and search for similar text.
- [Image Search](./examples/image_search/README.md): use the OpenAI CLIP model to generate vectors for image and text, store them in TiDB Vector, and search for similar images.
- [LlamaIndex RAG with UI](./examples/llamaindex-tidb-vector-with-ui/README.md): use the LlamaIndex to build an [RAG(Retrieval-Augmented Generation)](https://docs.llamaindex.ai/en/latest/getting_started/concepts/) application.
- [Chat with URL](./llamaindex-tidb-vector/README.md): use LlamaIndex to build an [RAG(Retrieval-Augmented Generation)](https://docs.llamaindex.ai/en/latest/getting_started/concepts/) application that can chat with a URL.
- [GraphRAG](./examples/graphrag-demo/README.md): 20 lines code of using TiDB Serverless to build a Knowledge Graph based RAG application.
- [GraphRAG Step by Step Tutorial](./examples/graphrag-step-by-step-tutorial/README.md): Step by step tutorial to build a Knowledge Graph based RAG application with Colab notebook. In this tutorial, you will learn how to extract knowledge from a text corpus, build a Knowledge Graph, store the Knowledge Graph in TiDB Serverless, and search from the Knowledge Graph.
- [Vector Search Notebook with SQLAlchemy](https://colab.research.google.com/drive/1LuJn4mtKsjr3lHbzMa2RM-oroUvpy83y?usp=sharing): use [SQLAlchemy](https://www.sqlalchemy.org/) to interact with TiDB Serverless: connect db, index&store data and then search vectors.
- [Build RAG with Jina AI Embeddings](./examples/jina-ai-embeddings-demo/README.md): use Jina AI to generate embeddings for text data, store the embeddings in TiDB Vector Storage, and search for similar embeddings.
- [Semantic Cache](./examples/semantic-cache/README.md): build a semantic cache with Jina AI and TiDB Vector.

for more examples, see the [examples](./examples) directory.

## Contributing
Please feel free to reach out to the maintainers if you have any questions or need help with the project. Before contributing, please read the [CONTRIBUTING.md](./CONTRIBUTING.md) file.