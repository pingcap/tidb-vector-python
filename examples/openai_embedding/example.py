import os
from openai import OpenAI
from peewee import Model, MySQLDatabase, TextField, SQL
from tidb_vector.peewee import VectorField

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
embedding_model = "text-embedding-3-small"
embedding_dimensions = 1536

db = MySQLDatabase(
   'test',
    user=os.environ.get('TIDB_USERNAME'),
    password=os.environ.get('TIDB_PASSWORD'),
    host=os.environ.get('TIDB_HOST'),
    port=4000,
    ssl_verify_cert=True,
    ssl_verify_identity=True
)

documents = [
   "TiDB is an open-source distributed SQL database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.",
   "TiFlash is the key component that makes TiDB essentially an Hybrid Transactional/Analytical Processing (HTAP) database. As a columnar storage extension of TiKV, TiFlash provides both good isolation level and strong consistency guarantee.",
   "TiKV is a distributed and transactional key-value database, which provides transactional APIs with ACID compliance. With the implementation of the Raft consensus algorithm and consensus state stored in RocksDB, TiKV guarantees data consistency between multiple replicas and high availability. ",
]

class DocModel(Model):
    text = TextField()
    embedding = VectorField(dimensions=embedding_dimensions)

    class Meta:
        database = db
        table_name = "openai_embedding_test"
    
    def __str__(self):
        return self.text

db.connect()
db.drop_tables([DocModel])
db.create_tables([DocModel])

embeddings = [
    r.embedding
    for r in client.embeddings.create(
      input=documents, model=embedding_model
    ).data
]
data_source = [
    {"text": doc, "embedding": emb}
    for doc, emb in zip(documents, embeddings)
]
DocModel.insert_many(data_source).execute()

question = "what is TiKV?"
question_embedding = client.embeddings.create(input=question, model=embedding_model).data[0].embedding
related_docs = DocModel.select(
    DocModel.text, DocModel.embedding.cosine_distance(question_embedding).alias("distance")
).order_by(SQL("distance")).limit(3)

print("Question:", question)
print("Related documents:")
for doc in related_docs:
    print(doc.distance, doc.text)

db.close()