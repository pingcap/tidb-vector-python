import os
import dotenv

from tidb_vector.peewee import VectorField, VectorAdaptor
from tidb_vector.constants import DistanceMetric
from peewee import Model, MySQLDatabase, TextField

dotenv.load_dotenv()

# Step 1: Connect to TiDB using Peewee.

# Using `pymysql` as the driver.
ssl_kwargs = {
    'ssl_verify_cert': True,
    'ssl_verify_identity': True,
}

# Using `mysqlclient` as the driver.
# ssl_kwargs = {
#     'ssl_mode': 'VERIFY_IDENTITY',
#     'ssl': {
#         # Root certificate default path
#         # https://docs.pingcap.com/tidbcloud/secure-connections-to-serverless-clusters/#root-certificate-default-path
#         'ca': os.environ.get('TIDB_CA_PATH', '/path/to/ca.pem'),
#     },
# }

db = MySQLDatabase(
    database=os.environ.get('TIDB_DATABASE', 'test'),
    user=os.environ.get('TIDB_USERNAME', 'root'),
    password=os.environ.get('TIDB_PASSWORD', ''),
    host=os.environ.get('TIDB_HOST', 'localhost'),
    port=int(os.environ.get('TIDB_PORT', '4000')),
    **ssl_kwargs if os.environ.get('TIDB_SSL', 'false').lower() == 'true' else {},
)


# Step 2: Define a table with a vector column.

# Create table without HNSW index.
class Document(Model):
    class Meta:
        database = db
        table_name = 'peewee_demo_documents'

    content = TextField()
    embedding = VectorField(3)


# Create table with HNSW index.
class DocumentWithIndex(Model):
    class Meta:
        database = db
        table_name = 'peewee_demo_documents_with_index'

    content = TextField()
    embedding = VectorField(3)


db.connect()
db.drop_tables([Document, DocumentWithIndex])
db.create_tables([Document, DocumentWithIndex])
VectorAdaptor(db).create_vector_index(
    DocumentWithIndex.embedding,
    DistanceMetric.COSINE,
)

# Step 3. Insert embeddings into the table.
Document.create(content='dog', embedding=[1, 2, 1])
Document.create(content='fish', embedding=[1, 2, 4])
Document.create(content='tree', embedding=[1, 0, 0])

# Step 4. Get the 3-nearest neighbor documents.
print('Get 3-nearest neighbor documents:')
distance = Document.embedding.cosine_distance([1, 2, 3]).alias('distance')
results = Document.select(Document, distance).order_by(distance).limit(3)

for doc in results:
    print(f'  - distance: {doc.distance}\n'
          f'    document: {doc.content}')

# Step 5. Get documents within a certain distance.
print('Get documents within a certain distance:')
distance_expression = Document.embedding.cosine_distance([1, 2, 3])
distance = distance_expression.alias('distance')
results = Document.select(Document, distance).where(distance_expression < 0.2).order_by(distance).limit(3)

for doc in results:
    print(f'  - distance: {doc.distance}\n'
          f'    document: {doc.content}')
