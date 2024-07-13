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


def get_embedding_column_definition(
    connection_string: str,
    table_name: str,
    column_name: str,
    engine_args: Optional[Dict[str, Any]] = None,
):
    """
    Retrieves the column definition of an embedding column from a database table.

    Args:
        connection_string (str): The connection string to the database.
        table_name (str): The name of the table.
        column_name (str): The name of the column.
        engine_args (Optional[Dict[str, Any]]): Additional arguments for the engine.

    Returns:
        tuple: A tuple containing the dimension (int or None) and distance metric (str or None).
    """
    engine = sqlalchemy.create_engine(connection_string, **(engine_args or {}))
    try:
        with engine.connect() as connection:
            query = f"""SELECT COLUMN_TYPE, COLUMN_COMMENT
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column_name}'"""
            result = connection.execute(sqlalchemy.text(query)).fetchone()
            if result:
                return extract_info_from_column_definition(result[0], result[1])
    finally:
        engine.dispose()

    return None, None


def extract_info_from_column_definition(column_type, column_comment):
    """
    Extracts the dimension and distance metric from a column definition,
    supporting both optional dimension and optional comment.

    Args:
        column_type (str): The column definition, possibly including dimension and a comment.

    Returns:
        tuple: A tuple containing the dimension (int or None) and the distance metric (str or None).
    """
    # Try to extract the dimension, which is optional.
    dimension_match = re.search(r"VECTOR(?:\((\d+)\))?", column_type, re.IGNORECASE)
    dimension = (
        int(dimension_match.group(1))
        if dimension_match and dimension_match.group(1)
        else None
    )

    # Extracting index type and distance metric from the comment, supporting both single and double quotes.
    distance_match = re.search(r"distance=([^,\)]+)", column_comment)
    distance = distance_match.group(1) if distance_match else None

    return dimension, distance
