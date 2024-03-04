import sqlalchemy
import re
from typing import Any, Dict, Optional


class EmbeddingColumnMismatchError(ValueError):
    """
    Exception raised when the existing embedding column does not match the expected dimension.

    Attributes:
        existing_col (str): The definition of the existing embedding column.
        expected_col (str): The definition of the expected embedding column.
    """

    def __init__(self, existing_col, expected_col):
        self.existing_col = existing_col
        self.expected_col = expected_col
        super().__init__(
            f"The existing embedding column ({existing_col}) does not match the expected dimension ({expected_col})."
        )


def check_table_existence(
    connection_string: str,
    table_name: str,
    engine_args: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Check if the vector table exists in the database

    Args:
        connection_string (str): The connection string for the database.
        table_name (str): The name of the table to check.
        engine_args (Optional[Dict[str, Any]]): Additional arguments for the engine.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    engine = sqlalchemy.create_engine(connection_string, **(engine_args or {}))
    try:
        inspector = sqlalchemy.inspect(engine)
        return table_name in inspector.get_table_names()
    finally:
        engine.dispose()


def get_embedding_column_definition(connection_string, table_name, column_name):
    """
    Retrieves the column definition of an embedding column from a database table.

    Args:
        connection_string (str): The connection string to the database.
        table_name (str): The name of the table.
        column_name (str): The name of the column.

    Returns:
        str: The column definition of the embedding column, or None if not found.
    """
    engine = sqlalchemy.create_engine(connection_string)
    try:
        with engine.connect() as connection:
            query = f"SELECT COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column_name}'"
            result = connection.execute(sqlalchemy.text(query)).fetchone()
            if result:
                return extract_dimension_from_column_definition(result[0])
    finally:
        engine.dispose()

    return None


def extract_dimension_from_column_definition(column_type):
    """
    Extracts the dimension from a column definition of type 'VECTOR<FLOAT>(dimension)'.

    Args:
        column_type (str): The column definition.

    Returns:
        int or None: The dimension if it exists, None otherwise.
    """
    match = re.search(r"VECTOR<FLOAT>(?:\((\d+)\))?", column_type, re.IGNORECASE)

    if match:
        if match.group(1):
            return int(match.group(1))
        return None
    return None
