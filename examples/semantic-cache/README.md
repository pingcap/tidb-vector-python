# Semantic Cache with Jina AI and TiDB Vector
Semantic cache is a cache that stores the semantic information of the data. It can be used to speed up the search process by storing the embeddings of the data and searching for similar embeddings. This example demonstrates how to use Jina AI to generate embeddings for text data and store the embeddings in TiDB Vector Storage. It also shows how to search for similar embeddings in TiDB Vector Storage.

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
cd tidb-vector-python/examples/semantic-cache
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables

Get the `HOST`, `PORT`, `USERNAME`, `PASSWORD`, and `DATABASE` from the TiDB Cloud console, as described in the [Prerequisites](../README.md#prerequisites) section. Then set the following environment variables:

```bash
export DATABASE_URI="mysql+pymysql://34u7xMnnDLSkjV1.root:<PASSWORD>@gateway01.eu-central-1.prod.aws.tidbcloud.com:4000/test?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=true&ssl_verify_identity=true"
```
or create a `.env` file with the above environment variables.


### Run this example


#### Start the semantic cache server
  
  ```bash
fastapi dev cache.py
  ```

#### Test the API

Get the Jina AI API key from the [Jina AI Embedding API](https://jina.ai/embeddings/) page, and save it somewhere safe for later use.

`POST /set`

```bash
curl --location ':8000/set' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer <your jina token>' \
--data '{
    "key": "what is tidb",
    "value": "tidb is a mysql-compatible and htap database"
}'
```

`GET /get/<key>`

```bash
curl --location ':8000/get/what%27s%20tidb%20and%20tikv?max_distance=0.5' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer <your jina token>'
```