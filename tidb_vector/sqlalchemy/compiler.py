from sqlalchemy.dialects.mysql.base import MySQLDDLCompiler
from sqlalchemy.sql import elements, operators, functions

from .ddl import CreateTiFlashReplica, CreateVectorIndex


class TiDBDDLCompiler(MySQLDDLCompiler):
    def visit_tiflash_replica(self, replica: CreateTiFlashReplica, **kw):
        # from IPython import embed;embed()
        return "ALTER TABLE {} SET TIFLASH REPLICA {}".format(
            replica.element.inner_table.name, replica.element.replica_num
        )

    def visit_create_vector_index(self, create: CreateVectorIndex, **kw):
        """Build the ``CREATE VECTOR INDEX ...`` statement
        MySQLDDLCompiler.visit_create_index
        """
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        table = preparer.format_table(index.table)

        columns = [
            self.sql_compiler.process(
                (
                    elements.Grouping(expr)
                    if (
                        isinstance(expr, elements.BinaryExpression)
                        or (
                            isinstance(expr, elements.UnaryExpression)
                            and expr.modifier
                            not in (operators.desc_op, operators.asc_op)
                        )
                        or isinstance(expr, functions.FunctionElement)
                    )
                    else expr
                ),
                include_table=False,
                literal_binds=True,
            )
            for expr in index.expressions
        ]

        name = self._prepared_index_name(index)

        text = "CREATE VECTOR INDEX "
        if create.if_not_exists:
            text += "IF NOT EXISTS "
        text += f"{name} ON {table} "

        text += f"({', '.join(columns)})"

        using = index.kwargs.get(f"{self.dialect.name}_using")
        if using is not None:
            text += f" USING {preparer.quote(using)}"

        return text
