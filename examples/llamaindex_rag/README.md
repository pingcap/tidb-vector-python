# LlamaIndex RAG Example

This example demonstrates how to use the LlamaIndex and TiDB Serverless to build a simple RAG(Retrival-Augmented Generation) application.

## Prerequisites

- A running TiDB Serverless cluster with vector search enabled
- Python 3.8 or later
- OpenAI [API key](https://platform.openai.com/docs/quickstart)

## Run the example

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables

Get the `OPENAI_API_KEY` from [OpenAI](https://platform.openai.com/docs/quickstart)

Get the `TIDB_HOST`, `TIDB_USERNAME`, and `TIDB_PASSWORD` from the TiDB Cloud console, as described in the [Prerequisites](../README.md#prerequisites) section.

```bash
export OPENAI_API_KEY="sk-*******"
export TIDB_HOST="gateway01.*******.shared.aws.tidbcloud.com"
export TIDB_USERNAME="****.root"
export TIDB_PASSWORD="****"
```

### Prepare data and run the server

```bash
# prepare the data
python example.py prepare

# runserver
python example.py runserver
```

Now you can visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to interact with the RAG application.
