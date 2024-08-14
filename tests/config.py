import os


class TestConfig:
    TIDB_HOST = os.getenv("TEST_TIDB_HOST", "127.0.0.1")
    TIDB_USER = os.getenv("TEST_TIDB_USER", "root")
    TIDB_PASSWORD = os.getenv("TEST_TIDB_PASSWORD", "")
    TIDB_PORT = int(os.getenv("TEST_TIDB_PORT", "4000"))
    TIDB_SSL = os.getenv("TEST_TIDB_SSL", "false").lower() in ["true", "1"]
