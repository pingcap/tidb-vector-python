# LlamaIndex RAG Example

This example demonstrates how to use the LlamaIndex and TiDB Serverless to build a simple RAG(Retrival-Augmented Generation) application. It crawl an example webpage and index the content to TiDB Serverless with LlamaIndex, then use the LlamaIndex to search the content and generate the answer with OpenAI.

## Prerequisites

- A running TiDB Serverless cluster with vector search enabled
- Python 3.8 or later
- OpenAI [API key](https://platform.openai.com/docs/quickstart)

## Run the example

### Clone this repo

```bash
git clone https://github.com/pingcap/tidb-vector-python.git
```

### Create a virtual environment

```bash
cd tidb-vector-python/examples/llamaindex-tidb-vector
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

### Run this example

```text
$ python chat_with_url.py --help
Usage: chat_with_url.py [OPTIONS]

Options:
  --url TEXT  URL you want to talk to,
              default=https://docs.pingcap.com/tidb/stable/overview
  --help      Show this message and exit.
$
$ python chat_with_url.py
Enter your question: : tidb vs mysql
TiDB is an open-source distributed SQL database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads. It is MySQL compatible and features horizontal scalability, strong consistency, and high availability. TiDB is designed to provide users with a one-stop database solution that covers OLTP, OLAP, and HTAP services. It offers easy horizontal scaling, financial-grade high availability, real-time HTAP capabilities, cloud-native features, and compatibility with the MySQL protocol and ecosystem.
Enter your question: :
```