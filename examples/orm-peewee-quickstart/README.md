# Integrate TiDB Vector Search with Peewee ORM

This is a simple demo to show how to integrate TiDB Vector Search with the Peewee ORM to search for similar text in a TiDB Serverless cluster.

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
cd tidb-vector-python/examples/orm-peewee-quickstart
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables

Create a `.env` file via the following command.

```shell
cp .env.example .env
```

Copy the `HOST`, `PORT`, `USERNAME`, `PASSWORD`, `DATABASE`, and `CA` parameters from the TiDB Cloud console (see [Prerequisites](../README.md#prerequisites)), and then set up the following environment variables in the `.env` file.

```bash
TIDB_HOST=gateway01.****.prod.aws.tidbcloud.com
TIDB_PORT=4000
TIDB_USERNAME=******.root
TIDB_PASSWORD=********
TIDB_DATABASE=test
# For macOS. For other platforms, please refer https://docs.pingcap.com/tidbcloud/secure-connections-to-serverless-clusters#root-certificate-default-path .
TIDB_CA_PATH=/etc/ssl/cert.pem
```

### Run this example

```text
$ python peewee-quickstart.py
Get 3-nearest neighbor documents:
  - distance: 0.00853986601633272
    document: fish
  - distance: 0.12712843905603044
    document: dog
  - distance: 0.7327387580875756
    document: tree
Get documents within a certain distance:
  - distance: 0.00853986601633272
    document: fish
  - distance: 0.12712843905603044
    document: dog
```