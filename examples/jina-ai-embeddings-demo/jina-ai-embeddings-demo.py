import os
import requests
import dotenv

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base
from tidb_vector.sqlalchemy import VectorType

dotenv.load_dotenv()


# Step 1. Define a helper function to generate embeddings using Jina AI's API.
JINAAI_API_KEY = os.getenv('JINAAI_API_KEY')
assert JINAAI_API_KEY is not None


def generate_embeddings(text: str):
    JINAAI_API_URL = 'https://api.jina.ai/v1/embeddings'
    JINAAI_HEADERS = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINAAI_API_KEY}'
    }
    JINAAI_REQUEST_DATA = {
        'input': [text],
        'model': 'jina-embeddings-v2-base-en'  # with dimisions 768.
    }
    response = requests.post(JINAAI_API_URL, headers=JINAAI_HEADERS, json=JINAAI_REQUEST_DATA)
    return response.json()['data'][0]['embedding']


# Step 2. Connect TiDB Serverless
TIDB_DATABASE_URL = os.getenv('TIDB_DATABASE_URL')
assert TIDB_DATABASE_URL is not None
engine = create_engine(url=TIDB_DATABASE_URL, pool_recycle=300)


# Step 3. Create the vector table.
Base = declarative_base()


class Document(Base):
    __tablename__ = "jinaai_tidb_demo_documents"

    id = Column(Integer, primary_key=True)
    content = Column(String(255), nullable=False)
    content_vec = Column(
        # DIMENSIONS is determined by the embedding model,
        # for Jina AI's jina-embeddings-v2-base-en model it's 768.
        VectorType(dim=768),
        comment="hnsw(distance=cosine)"
    )


Base.metadata.create_all(engine)


# Step 4. Generate embeddings for texts via Jina AI API and store them in TiDB.

TEXTS = [
    'Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.',
    'TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.',
]

data = []
for text in TEXTS:
    # Generate the embedding for the text via Jina AI API.
    embedding = generate_embeddings(text)
    data.append({
        'text': text,
        'embedding': embedding
    })

with Session(engine) as session:
    print('- Inserting Data to TiDB...')
    for item in data:
        print(f'  - Inserting: {item["text"]}')
        session.add(Document(
            content=item['text'],
            content_vec=item['embedding']
        ))
    session.commit()


# Step 5. Query the most relevant document based on the query.
query = 'What is TiDB?'
# Generate the embedding for the query via Jina AI API.
query_embedding = generate_embeddings(query)
with Session(engine) as session:
    print('- List All Documents and Their Distances to the Query:')
    for doc, distance in session.query(
        Document,
        Document.content_vec.cosine_distance(query_embedding).label('distance')
    ).all():
        print(f'  - distance: {distance}\n'
              f'    content: {doc.content}')

    print('- The Most Relevant Document and Its Distance to the Query:')
    doc, distance = session.query(
        Document,
        Document.content_vec.cosine_distance(query_embedding).label('distance')
    ).order_by(
        'distance'
    ).limit(1).first()
    print(f'  - distance: {distance}\n'
          f'    content: {doc.content}')

# Expected Output:
#
# - Inserting Data to TiDB...
#   - Inserting: Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.
#   - Inserting: TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.
# - List All Documents and Their Distances to the Query:
#   - distance: 0.3585317326132522
#     content: Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.
#   - distance: 0.10858102967720984
#     content: TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.
# - The Most Relevant Document and Its Distance to the Query:
#   - distance: 0.10858102967720984
#     content: TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.

