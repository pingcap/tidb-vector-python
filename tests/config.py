import os


class TestConfig:
    TIDB_HOST = os.getenv("TEST_TIDB_HOST", "")
    TIDB_USER = os.getenv("TEST_TIDB_USER", "root")
    TIDB_PASSWORD = os.getenv("TEST_TIDB_PASSWORD", "")
    TIDB_PORT = int(os.getenv("TEST_TIDB_PORT", "4000"))
