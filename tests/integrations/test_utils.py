"""Test TiDB Vector Search functionality."""
from __future__ import annotations

from tidb_vector.integrations.utils import extract_dimension_from_column_definition


def test_extract_dimension_from_column_definition():
    # Test case 1: column_type with dimension
    column_type = "vector<float>(1536) DEFAULT NULL"
    assert extract_dimension_from_column_definition(column_type) == 1536

    # Test case 2: column_type without dimension
    column_type = "vector<float> DEFAULT NULL"
    assert extract_dimension_from_column_definition(column_type) is None

    # Test case 3: column_type with invalid dimension
    column_type = "vector<float>(abc) DEFAULT NULL"
    assert extract_dimension_from_column_definition(column_type) is None

    # Test case 4: column_type with negative dimension
    column_type = "VECTOR<FLOAT>(-5)"
    assert extract_dimension_from_column_definition(column_type) is None
