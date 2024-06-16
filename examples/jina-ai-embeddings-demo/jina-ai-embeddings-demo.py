import os
import requests
import dotenv

from sqlalchemy import Column, Integer, String, create_engine, URL
from sqlalchemy.orm import Session, declarative_base
from tidb_vector.sqlalchemy import VectorType

dotenv.load_dotenv()

JINAAI_API_KEY = os.getenv('JINAAI_API_KEY')
assert JINAAI_API_KEY is not None
TIDB_USERNAME = os.getenv('TIDB_USERNAME')
TIDB_PASSWORD = os.getenv('TIDB_PASSWORD')
TIDB_HOST = os.getenv('TIDB_HOST')
TIDB_PORT = os.getenv('TIDB_PORT')
TIDB_DATABASE = os.getenv('TIDB_DATABASE')
assert TIDB_USERNAME is not None
assert TIDB_PASSWORD is not None
assert TIDB_HOST is not None
assert TIDB_PORT is not None
assert TIDB_DATABASE is not None

TEXTS = [
    'Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.',
    'TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.',
]

# 1. Get Embeddings from Jina AI
def generate_embeddings(text: str):
    JINAAI_API_URL = 'https://api.jina.ai/v1/embeddings'
    JINAAI_HEADERS = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINAAI_API_KEY}'
    }
    JINAAI_REQUEST_DATA = {
        'input': [text],
        'model': 'jina-embeddings-v2-base-en'  # with dimisions 768
    }
    response = requests.post(JINAAI_API_URL, headers=JINAAI_HEADERS, json=JINAAI_REQUEST_DATA)
    return response.json()['data'][0]['embedding']

data = []
for text in TEXTS:
    embedding = generate_embeddings(text)
    data.append({
        'text': text,
        'embedding': embedding
    })


# 2. Connect TiDB Serverless and Create Table
url = URL(
    drivername="mysql+pymysql",
    username=TIDB_USERNAME,
    password=TIDB_PASSWORD,
    host=TIDB_HOST,
    port=int(TIDB_PORT),
    database=TIDB_DATABASE,
    query={"ssl_verify_cert": True, "ssl_verify_identity": True},
)
engine = create_engine(url, pool_recycle=300)
Base = declarative_base()

class Document(Base):
    __tablename__ = "jinaai_tidb_demo_documents"

    id = Column(Integer, primary_key=True)
    content = Column(String(255), nullable=False)
    content_vec = Column(
        # DIMENSIONS is determined by the embedding model,
        # for Jina AI's jina-embeddings-v2-base-en model it's 768
        VectorType(dim=768),
        comment="hnsw(distance=l2)"
    )
# Create the table
Base.metadata.create_all(engine)


# 3. Insert Data from Jina AI to TiDB
with Session(engine) as session:
    print('- Inserting Data to TiDB...')
    for item in data:
        print(f'  - Inserting: {item["text"]}')
        session.add(Document(
            content=item['text'],
            content_vec=item['embedding']
        ))
    session.commit()


# 4. Query Data from TiDB
query = 'What is TiDB?'
query_embedding = generate_embeddings(query)
with Session(engine) as session:
    print('- List All Documents and Their Distances to the Query:')
    for doc, distance in session.query(
        Document,
        Document.content_vec.cosine_distance(query_embedding).label('distance')
    ).all():
        print(f'  - {doc.content}: {distance}')

    print('- The Most Relevant Document and Its Distance to the Query:')
    doc, distance = session.query(
        Document,
        Document.content_vec.cosine_distance(query_embedding).label('distance')
    ).order_by(
        'distance'
    ).limit(1).first()
    print(f'  - {doc.content}: {distance}')

# Output:
#
# - Inserting Data to TiDB...
#   - Inserting: Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.
#   - Inserting: TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.
# - List All Documents and Their Distances to the Query:
#   - Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.: 0.3585317326132522
#   - TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.: 0.10858658947444844
# - The Most Relevant Document and Its Distance to the Query:
#   - TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.: 0.10858658947444844