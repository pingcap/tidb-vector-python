import os
from datetime import datetime
from typing import Optional, Annotated

import requests
import dotenv
from fastapi import Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import (
    SQLModel,
    Session,
    create_engine,
    select,
    Field,
    Column,
    String,
    Text,
    DateTime,
)
from sqlalchemy import func
from tidb_vector.sqlalchemy import VectorType
dotenv.load_dotenv()


# Configuration from .env
# Example: "mysql+pymysql://<username>:<password>@<host>:<port>/<database>?ssl_mode=VERIFY_IDENTITY&ssl_ca=/etc/ssl/cert.pem"
DATABASE_URI = os.getenv('DATABASE_URI')
# Ref: https://docs.pingcap.com/tidb/stable/time-to-live
# Default: 604800 SECOND (1 week)
TIME_TO_LIVE = os.getenv('TIME_TO_LIVE')


# Get Embeddings from Jina AI
def generate_embeddings(jinaai_api_key: str, text: str):
    JINAAI_API_URL = 'https://api.jina.ai/v1/embeddings'
    JINAAI_HEADERS = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {jinaai_api_key}'
    }
    JINAAI_REQUEST_DATA = {
        'input': [text],
        'model': 'jina-embeddings-v2-base-en'  # with dimisions 768
    }
    response = requests.post(JINAAI_API_URL, headers=JINAAI_HEADERS, json=JINAAI_REQUEST_DATA)
    return response.json()['data'][0]['embedding']


class Cache(SQLModel, table=True):
    __table_args__ = {
        'mysql_TTL': f'created_at + INTERVAL {TIME_TO_LIVE} SECOND',
    }

    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(sa_column=Column(String(255), unique=True, nullable=False))
    key_vec: Optional[list[float]]= Field(
        sa_column=Column(
            VectorType(768),
            default=None,
            comment="hnsw(distance=l2)",
            nullable=False,
        )
    )
    value: Optional[str] = Field(sa_column=Column(Text))
    created_at: datetime = Field(
        sa_column=Column(DateTime, server_default=func.now(), nullable=False)
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
        )
    )

engine = create_engine(DATABASE_URI)
SQLModel.metadata.create_all(engine)

app = FastAPI()
security = HTTPBearer()

readme = open("README.md").read()

@app.get("/")
async def index():
    return {
        "message": "Welcome to Semantic Cache API, it is built using Jina AI Embeddings API and TiDB Vector",
        "docs": "/docs",
        "redoc": "/redoc",
        "about": "https://github.com/pingcap/tidb-vector-python/blob/main/examples/semantic-cache/README.md",
        "config": {
            "TIME_TO_LIVE": int(TIME_TO_LIVE),
            "EMBEDDING_DIMENSIONS": 768,
            "EMBEDDING_PROVIDER": "Jina AI",
            "EMBEDDING_MODEL": "jina-embeddings-v2-base-en",
        }
    }


# /set method of Semantic Cache
@app.post("/set")
async def set(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    cache: Cache,
):
    cache.key_vec = generate_embeddings(credentials.credentials, cache.key)

    with Session(engine) as session:
        session.add(cache)
        session.commit()

    return {'message': 'Cache has been set'}


@app.get("/get/{key}")
async def get(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    key: str,
    max_distance: Optional[float] = 0.1,
):
    key_vec = generate_embeddings(credentials.credentials, key)
    # The max value of distance is 0.3
    max_distance = min(max_distance, 0.3)

    with Session(engine) as session:
        result = session.exec(
            select(
                Cache,
                Cache.key_vec.cosine_distance(key_vec).label('distance')
            ).filter(
                Cache.key_vec.cosine_distance(key_vec) <= max_distance
            ).order_by(
                'distance'
            ).limit(1)
        ).first()

        if result is None:
            return {"message": "Cache not found"}, 404
        else:
            cache, distance = result
            return {
                "key": cache.key,
                "value": cache.value,
                "distance": distance
            }