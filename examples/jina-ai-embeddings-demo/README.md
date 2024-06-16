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

Get the `TIDB_HOST`, `TIDB_USERNAME`, `TIDB_PASSWORD`, `TIDB_HOST`, `TIDB_PORT` and `TIDB_DATABASE` from the TiDB Cloud console, as described in the [Prerequisites](../README.md#prerequisites) section.

```bash
export JINA_API_KEY="****"
export TIDB_HOST="gateway01.*******.shared.aws.tidbcloud.com"
export TIDB_USERNAME="****.root"
export TIDB_PASSWORD="****"
export TIDB_PORT="4000"
export TIDB_DATABASE="test"
```
or create a `.env` file with the above environment variables.


### Run this example

```text
$ python jina-ai-embeddings-demo.py
- Inserting Data to TiDB...
  - Inserting: Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.
  - Inserting: TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.
- List All Documents and Their Distances to the Query:
  - Jina AI offers best-in-class embeddings, reranker and prompt optimizer, enabling advanced multimodal AI.: 0.3585317326132522
  - TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.: 0.10858658947444844
- The Most Relevant Document and Its Distance to the Query:
  - TiDB is an open-source MySQL-compatible database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.: 0.10858658947444844
```