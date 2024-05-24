# LlamaIndex RAG Example

This example demonstrates how to use the DSPy and TiDB Serverless to build a simple GraphRAG application. It crawled an example webpage and index the content to TiDB Serverless with Graph, then use the Graph and Vector to search the content and generate the answer with OpenAI.

## Prerequisites

- A running TiDB Serverless cluster
  - Vector search enabled
  - Run the [init.sql](./init.sql) in your cluster
- Python 3.8 or later
- OpenAI [API key](https://platform.openai.com/docs/quickstart)

## Run the example

### Clone this repo

```bash
git clone https://github.com/pingcap/tidb-vector-python.git
```

### Create a virtual environment

```bash
cd tidb-vector-python/examples/graphrag-simple-retrieve
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables

Get the TiDB connection string via `TIDB_HOST`, `TIDB_USERNAME`, and `TIDB_PASSWORD` from the TiDB Cloud console, as described in the [Prerequisites](../README.md#prerequisites) section.

The TiDB connection string will look like:

```
mysql+pymysql://{TIDB_USER}:{TIDB_PASSWORD}@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DB_NAME}?ssl_verify_cert=True&ssl_verify_identity=True
```

Get the `OPENAI_API_KEY` from [OpenAI](https://platform.openai.com/docs/quickstart)

### Run this example


```text
$ python3 simple-retrieve.py
Input your TIDB connection string:
Input your OpenAI API Key:
Enter your question:
```