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

# VectorIndex.argument_for("mysql", "add_tiflash_on_demand", None)

# Table.argument_for("mysql", "tiflash", None)

@compiles(sqlalchemy.schema.CreateTable)
def compile_create_table(create_table_elem: sqlalchemy.sql.ddl.CreateTable, compiler: sqlalchemy.sql.compiler.DDLCompiler, **kw):
    text = compiler.visit_create_table(create_table_elem, **kw)
    # table_elem = create_table_elem.element
    # if table_elem.dialect_options.get("mysql", {}).get("tiflash_replica"):
    #     text += " TIFLASH_REPLICA = 1"
    return text
