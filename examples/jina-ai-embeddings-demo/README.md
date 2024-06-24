# Jina AI Embeddings Demo
This is a simple demo to show how to use Jina AI to generate embeddings for text data. Then store the embeddings in TiDB Vector Storage and search for similar embeddings.

## Prerequisites

- A running TiDB Serverless cluster with vector search enabled
- Python 3.8 or later
- Jina AI API key

## Run the example

### Clone this repo

```bash
git clone https://github.com/pingcap/tidb-vector-python.git
```

### Create a virtual environment

```bash
cd tidb-vector-python/examples/jina-ai-embeddings-demo
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables

Get the Jina AI API key from the [Jina AI Embedding API](https://jina.ai/embeddings/) page

Get the `HOST`, `PORT`, `USERNAME`, `PASSWORD`, `DATABASE`, and `CA` parameters from the TiDB Cloud console (see [Prerequisites](../README.md#prerequisites)), and then replace the following placeholders to get the `TIDB_DATABASE_URL`.

```bash
export JINA_API_KEY="****"
export TIDB_DATABASE_URL="mysql+pymysql://<USERNAME>:<PASSWORD>@<HOST>:4000/<DATABASE>?ssl_ca=<CA>&ssl_verify_cert=true&ssl_verify_identity=true"
```
or create a `.env` file with the above environment variables.


### Run this example

```text
$ python jina-ai-embeddings-demo.py
- Inserting Data to TiDB...
  - Inserting: Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.
  - Inserting: TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.
- List All Documents and Their Distances to the Query:
  - distance: 0.3585317326132522
    content: Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.
  - distance: 0.10858102967720984
    content: TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.
- The Most Relevant Document and Its Distance to the Query:
  - distance: 0.10858102967720984
    content: TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.
```