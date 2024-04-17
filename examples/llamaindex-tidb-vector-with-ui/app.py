import os
import sys
import json
import logging
import click
import uvicorn
import fastapi
import asyncio
from enum import Enum
from sqlalchemy import URL
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.base.response.schema import StreamingResponse as llamaStreamingResponse
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.readers.web import SimpleWebPageReader


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class EventType(Enum):
    META = 1
    ANSWER = 2


logger.info("Initializing TiDB Vector Store....")
tidb_connection_url = URL(
    "mysql+pymysql",
    username=os.environ['TIDB_USERNAME'],
    password=os.environ['TIDB_PASSWORD'],
    host=os.environ['TIDB_HOST'],
    port=4000,
    database="test",
    query={"ssl_verify_cert": True, "ssl_verify_identity": True},
)
tidbvec = TiDBVectorStore(
    connection_string=tidb_connection_url,
    table_name="llama_index_rag_test",
    distance_strategy="cosine",
    vector_dimension=1536, # Length of the vectors returned by the model
    drop_existing_table=False,
)
tidb_vec_index = VectorStoreIndex.from_vector_store(tidbvec)
storage_context = StorageContext.from_defaults(vector_store=tidbvec)
query_engine = tidb_vec_index.as_query_engine(streaming=True)
logger.info("TiDB Vector Store initialized successfully")


def do_prepare_data():
    logger.info("Preparing the data for the application")
    documents = SimpleWebPageReader(html_to_text=True).load_data(
        ["http://paulgraham.com/worked.html"]
    )
    tidb_vec_index.from_documents(documents, storage_context=storage_context, show_progress=True)
    logger.info("Data preparation complete")


# https://stackoverflow.com/questions/76288582/is-there-a-way-to-stream-output-in-fastapi-from-the-response-i-get-from-llama-in
async def astreamer(response: llamaStreamingResponse):
    try:
        meta = json.dumps(jsonable_encoder(list(vars(node) for node in response.source_nodes)))
        yield f'{EventType.META.value}: {meta}\n\n'
        for i in response.response_gen:
            yield f'{EventType.ANSWER.value}: {i}\n\n'
            await asyncio.sleep(.1)
    except asyncio.CancelledError as e:
        print('cancelled')


app = fastapi.FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get('/', response_class=HTMLResponse)
def index(request: fastapi.Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/ask')
async def ask(q: str):
    response = query_engine.query(q)
    return StreamingResponse(astreamer(response), media_type='text/event-stream')


@click.group(context_settings={'max_content_width': 150})
def cli():
    pass


@cli.command()
@click.option('--host', default='127.0.0.1', help="Host, default=127.0.0.1")
@click.option('--port', default=3000, help="Port, default=3000")
@click.option('--reload', is_flag=True, help="Enable auto-reload")
def runserver(host, port, reload):
    uvicorn.run(
        "__main__:app", host=host, port=port, reload=reload,
        log_level="debug", workers=1,
    )


@cli.command()
def prepare():
    do_prepare_data()


if __name__ == '__main__':
    cli()
