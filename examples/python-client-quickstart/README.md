# TiDB Vector Search Python Client Quickstart

This is a simple demo to show how to use the TiDB Vector Search Python Client to search for similar text in a TiDB Serverless cluster.

## Prerequisites

- A running TiDB Serverless cluster with vector search enabled
- Python 3.8 or later

## Run the example

### Clone this repo

```bash
git clone https://github.com/pingcap/tidb-vector-python.git
```

### Create a virtual environment

```bash
cd tidb-vector-python/examples/python-client-quickstart
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables

Get the `HOST`, `PORT`, `USERNAME`, `PASSWORD`, `DATABASE`, and `CA` parameters from the TiDB Cloud console (see [Prerequisites](../README.md#prerequisites)), and then replace the following placeholders to get the `TIDB_DATABASE_URL`.

```bash
export TIDB_DATABASE_URL="mysql+pymysql://<USERNAME>:<PASSWORD>@<HOST>:4000/<DATABASE>?ssl_ca=<CA>&ssl_verify_cert=true&ssl_verify_identity=true"
```
or create a `.env` file with the above environment variables.

### Run this example

```text
$ python example.py
Downloading and loading the embedding model...
Search result ("a swimming animal"):
- text: "fish", distance: 0.4562914811223072
- text: "dog", distance: 0.6469335836410557
- text: "tree", distance: 0.798545178640937
```