# GeminiAI Embedding Example

This example demonstrates how to utilize GeminiAI embedding for semantic search. According to GeminiAI's [documentation](https://ai.google.dev/gemini-api/docs/embeddings), we will use cosine similarity to calculate vector distance.

You can run this example in two ways:

- [Run in Jupyter Notebook](#jupyter-notebook)
- [Run in Local](#run-in-local)

## Jupyter Notebook

Notebook: [example.ipynb](./example.ipynb)

Try it in the [Google colab](https://colab.research.google.com/github/pingcap/tidb-vector-python/blob/main/examples/gemini-ai-embeddings-demo/example.ipynb).

## Run in Local

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install the requirements

```bash
pip install -r requirements.txt
```

### Set the environment variables

Get the `GEMINI_API_KEY` from [GeminiAI](https://ai.google.dev/gemini-api/docs/quickstart)

Get the `TIDB_HOST`, `TIDB_USERNAME`, and `TIDB_PASSWORD` from the TiDB Cloud console, as described in the [Prerequisites](../README.md#prerequisites) section.

```bash
export GEMINI_API_KEY="*******"
export TIDB_HOST="gateway01.*******.shared.aws.tidbcloud.com"
export TIDB_USERNAME="****.root"
export TIDB_PASSWORD="****"
```

### Run the example

```bash
python3 example.py
```
