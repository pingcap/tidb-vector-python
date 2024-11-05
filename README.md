# tidb-vector-python

Use TiDB Vector Search with Python.

## Installation

```bash
pip install tidb-vector
```

## Usage

TiDB is a SQL database so that this package introduces Vector Search capability for Python ORMs:

- [#SQLAlchemy](#sqlalchemy)
- [#Django](#django)
- [#Peewee](#peewee)

Pick one that you are familiar with to get started. If you are not using any of them, we recommend [#SQLAlchemy](#sqlalchemy).

We also provide a Vector Search client for simple usage:

- [#TiDB Vector Client](#tidb-vector-client)

### SQLAlchemy

```bash
pip install tidb-vector sqlalchemy pymysql
```

```python
from sqlalchemy import Integer, Text, Column
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, declarative_base

import tidb_vector
from tidb_vector.sqlalchemy import VectorType, VectorAdaptor

engine = create_engine("mysql+pymysql://root@127.0.0.1:4000/test")
Base = declarative_base()


# Define table schema
class Doc(Base):
    __tablename__ = "doc"
    id = Column(Integer, primary_key=True)
    embedding = Column(VectorType(3)) # Vector with 3 dimensions
    content = Column(Text)


# Create empty table
Base.metadata.drop_all(engine)  # clean data from last run
Base.metadata.create_all(engine)

# Create index using L2 distance
adaptor = VectorAdaptor(engine)
adaptor.create_vector_index(
    Doc.embedding, tidb_vector.DistanceMetric.L2, skip_existing=True
)

# Insert content with vectors
with Session(engine) as session:
    session.add(Doc(id=1, content="dog", embedding=[1, 2, 1]))
    session.add(Doc(id=2, content="fish", embedding=[1, 2, 4]))
    session.add(Doc(id=3, content="tree", embedding=[1, 0, 0]))
    session.commit()

# Perform Vector Search for Top K=1
with Session(engine) as session:
    results = session.execute(
        select(Doc.id, Doc.content)
        .order_by(Doc.embedding.cosine_distance([1, 2, 3]))
        .limit(1)
    ).all()
    print(results)

# Perform filtered Vector Search by adding a Where Clause:
with Session(engine) as session:
    results = session.execute(
        select(Doc.id, Doc.content)
        .where(Doc.id > 2)
        .order_by(Doc.embedding.cosine_distance([1, 2, 3]))
        .limit(1)
    ).all()
    print(results)
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

Within the framework, you can directly utilize the built-in `TiDBVectorClient`, as demonstrated by integrations like [Langchain](https://python.langchain.com/docs/integrations/vectorstores/tidb_vector) and [Llama index](https://docs.llamaindex.ai/en/stable/community/integrations/vector_stores.html#using-a-vector-store-as-an-index), to seamlessly interact with TiDB Vector. This approach abstracts away the need to manage the underlying ORM, simplifying your interaction with the vector store.

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
