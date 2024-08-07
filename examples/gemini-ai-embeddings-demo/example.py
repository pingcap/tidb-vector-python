import os
from peewee import Model, MySQLDatabase, TextField, SQL
from tidb_vector.peewee import VectorField
import google.generativeai as genai # Hypothetical import for Gemini API client

# Init Gemini client
# Adjust the initialization according to the Gemini API documentation
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
embedding_model = 'models/embedding-001' # Replace with the actual model name
embedding_dimensions = 768  # Adjust if different for the Gemini model

# Init TiDB connection
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
    "TiKV is a distributed and transactional key-value database, which provides transactional APIs with ACID compliance. With the implementation of the Raft consensus algorithm and consensus state stored in RocksDB, TiKV guarantees data consistency between multiple replicas and high availability.",
]

# Define a model with a VectorField to store the embeddings
class DocModel(Model):
    text = TextField()
    embedding = VectorField(dimensions=embedding_dimensions)

    class Meta:
        database = db
        table_name = "gemini_embedding_test"

    def __str__(self):
        return self.text

db.connect()
db.drop_tables([DocModel])
db.create_tables([DocModel])

# Insert the documents and their embeddings into TiDB
embeddings = genai.embed_content(model=embedding_model, content=documents, task_type="retrieval_document")
data_source = [
    {"text": doc, "embedding": emb}
    for doc, emb in zip(documents, embeddings['embedding'])
]
DocModel.insert_many(data_source).execute()

# Query the most similar documents to a question
# 1. Generate the embedding of the question
# 2. Query the most similar documents based on the cosine distance in TiDB
# 3. Print the results
question = "what is TiKV?"
question_embedding = genai.embed_content(model=embedding_model, content=[question], task_type="retrieval_query")['embedding'][0]
related_docs = DocModel.select(
    DocModel.text, DocModel.embedding.cosine_distance(question_embedding).alias("distance")
).order_by(SQL("distance")).limit(3)

print("Question:", question)
print("Related documents:")
for doc in related_docs:
    print(doc.distance, doc.text)

db.close()

# Expected Output:
# 
# Question: what is TiKV?
# Related documents:
# 0.22371791507562544 TiKV is a distributed and transactional key-value database, which provides transactional APIs with ACID compliance. With the implementation of the Raft consensus algorithm and consensus state stored in RocksDB, TiKV guarantees data consistency between multiple replicas and high availability. 
# 0.3317073143109729 TiFlash is the key component that makes TiDB essentially an Hybrid Transactional/Analytical Processing (HTAP) database. As a columnar storage extension of TiKV, TiFlash provides both good isolation level and strong consistency guarantee.
# 0.3690570695898543 TiDB is an open-source distributed SQL database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.