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
        # add tiflash automatically when creating vector index
        self.dialect_options["mysql"]["add_tiflash_on_demand"] = True

# VectorIndex.argument_for("mysql", "add_tiflash_on_demand", None)

@compiles(sqlalchemy.schema.CreateIndex)
def compile_create_vector_index(create_index_elem: sqlalchemy.sql.ddl.CreateIndex, compiler: sqlalchemy.sql.compiler.DDLCompiler, **kw):
    text = compiler.visit_create_index(create_index_elem, **kw)
    index_elem = create_index_elem.element
    if index_elem.dialect_options.get("mysql", {}).get("add_tiflash_on_demand"):
        text += " ADD_TIFLASH_ON_DEMAND"
    return text
