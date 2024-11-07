from typing import Optional, Any

import sqlalchemy

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import Index

class VectorIndex(Index):
    def __init__(
        self,
        name: Optional[str],
        *expressions, # _DDLColumnArgument
        _table: Optional[Any] = None,
        **dialect_kw: Any,
    ):
        super().__init__(name, *expressions, unique=False, _table=_table, **dialect_kw)
        self.dialect_options["mysql"]["prefix"] = "VECTOR"
