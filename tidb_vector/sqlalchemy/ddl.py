from typing import Any, Optional, Sequence, Union

import sqlalchemy
from sqlalchemy.sql.ddl import SchemaGenerator as SchemaGeneratorBase
from sqlalchemy.sql.ddl import SchemaDropper as SchemaDropperBase
from sqlalchemy.sql.base import DialectKWArgs
from sqlalchemy.sql.schema import (
    ColumnCollectionMixin,
    HasConditionalDDL,
    SchemaItem,
    MetaData as MetaDataBase,
    Table as TableBase,
    ColumnElement,
)
import sqlalchemy.exc as exc


class MetaData(MetaDataBase):
    """
    A collection of :class:`.Table` objects and their associated schema constructs.
    Overwrites the default implementation to use :class:`TiDBSchemaGenerator` and :class:`TiDBSchemaDropper`.
    """

    def create_all(self, bind, tables=None, checkfirst: bool = True) -> None:
        bind._run_ddl_visitor(
            TiDBSchemaGenerator, self, checkfirst=checkfirst, tables=tables
        )

    def drop_all(self, bind, tables=None, checkfirst: bool = True) -> None:
        bind._run_ddl_visitor(
            TiDBSchemaDropper, self, checkfirst=checkfirst, tables=tables
        )


class Table(TableBase):
    """
    Represent a table in a database.
    Overwrites the default implementation to use :class:`TiDBSchemaGenerator` and :class:`TiDBSchemaDropper`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create(self, bind, checkfirst: bool = False) -> None:
        """Issue a ``CREATE`` statement for this
        :class:`_schema.Table`, using the given
        :class:`.Connection` or :class:`.Engine`
        for connectivity.

        Overwrites the default implementation to use :class:`TiDBSchemaGenerator`
        """
        bind._run_ddl_visitor(TiDBSchemaGenerator, self, checkfirst=checkfirst)

    def drop(self, bind, checkfirst: bool = False) -> None:
        """Issue a ``DROP`` statement for this
        :class:`_schema.Table`, using the given
        :class:`.Connection` or :class:`.Engine` for connectivity.

        Overwrites the default implementation to use :class:`TiDBSchemaDropper`

        """
        bind._run_ddl_visitor(TiDBSchemaDropper, self, checkfirst=checkfirst)


class TiFlashReplica(DialectKWArgs, SchemaItem):
    """Represent the tiflash replica table attribute"""

    __visit_name__ = "tiflash_replica"

    inner_table: Optional[Table]

    @property
    def bind(self):
        return self.metadata.bind

    @property
    def metadata(self):
        return self.inner_table.metadata

    def __init__(self, inner_table: Table, num=1, *args, **kwargs) -> None:
        super().__init__()
        self.inner_table = inner_table
        self.replica_num = num
        # set the metadata to the inner_table
        self.inner_table.info["has_tiflash_replica"] = True

    def create(self, bind=None):
        """Issue a ``ALTER TABLE ... SET TIFLASH REPLICA {num}`` statement"""
        if bind is None:
            bind = self.bind
        bind._run_ddl_visitor(TiDBSchemaGenerator, self)

    def drop(self, bind=None):
        """Issue a ``ALTER TABLE ... SET TIFLASH REPLICA 0`` statement"""
        if bind is None:
            bind = self.bind
        bind._run_ddl_visitor(TiDBSchemaDropper, self)
        self.replica_num = 0


class VectorIndex(DialectKWArgs, ColumnCollectionMixin, HasConditionalDDL, SchemaItem):
    __visit_name__ = "vector_index"

    table: Optional[Table]
    expressions: Sequence[Union[str, ColumnElement[Any]]]
    _table_bound_expressions: Sequence[ColumnElement[Any]]

    @property
    def bind(self):
        return self.metadata.bind

    @property
    def metadata(self):
        return self.table.metadata

    def __init__(
        self,
        name: Optional[str],
        expressions,
        _table: Optional[Table] = None,
    ) -> None:
        super().__init__()
        self.table = table = None
        if _table is not None:
            table = _table

        self.name = name

        self.expressions = []
        # will call _set_parent() if table-bound column
        # objects are present
        ColumnCollectionMixin.__init__(
            self,
            expressions,
            _column_flag=False,
            _gather_expressions=self.expressions,
        )
        if table is not None:
            self._set_parent(table)

    def _set_parent(self, parent, **kw: Any) -> None:
        table = parent
        assert isinstance(table, Table)
        ColumnCollectionMixin._set_parent(self, table)

        if self.table is not None and table is not self.table:
            raise exc.ArgumentError(
                f"Index '{self.name}' is against table "
                f"'{self.table.description}', and "
                f"cannot be associated with table '{table.description}'."
            )
        self.table = table
        table.indexes.add(self)

        expressions = self.expressions
        col_expressions = self._col_expressions(table)
        assert len(expressions) == len(col_expressions)

        exprs = []
        for expr, colexpr in zip(expressions, col_expressions):
            if isinstance(expr, sqlalchemy.sql.ClauseElement):
                exprs.append(expr)
            elif colexpr is not None:
                exprs.append(colexpr)
            else:
                assert False
        self.expressions = self._table_bound_expressions = exprs

    def create(self, bind, checkfirst: bool = False) -> None:
        """Issue a ``CREATE`` statement for this
        :class:`.VectorIndex`, using the given
        :class:`.Connection` or :class:`.Engine`` for connectivity.
        """
        if bind is None:
            bind = self.bind
        bind._run_ddl_visitor(TiDBSchemaGenerator, self, checkfirst=checkfirst)

    def drop(self, bind, checkfirst: bool = False) -> None:
        """Issue a ``DROP`` statement for this
        :class:`.VectorIndex`, using the given
        :class:`.Connection` or :class:`.Engine` for connectivity.
        """
        if bind is None:
            bind = self.bind
        bind._run_ddl_visitor(TiDBSchemaDropper, self, checkfirst=checkfirst)


class AlterTiFlashReplica(sqlalchemy.sql.ddl._CreateDropBase):
    """Represent a ``ALTER TABLE ... SET TIFLASH REPLICA ...`` statement."""

    __visit_name__: str = "alter_tiflash_replica"

    def __init__(self, element, new_num):
        super().__init__(element)
        self.new_num = new_num


class CreateVectorIndex(sqlalchemy.sql.ddl.CreateIndex):
    """Represent a ``CREATE VECTOR INDEX ... ON ...`` statement."""

    __visit_name__: str = "create_vector_index"

    def __init__(self, element, if_not_exists=False):
        super().__init__(element, if_not_exists)


class TiDBSchemaGenerator(SchemaGeneratorBase):
    """Building logical CERATE ... statements."""

    def __init__(self, dialect, connection, checkfirst=False, tables=None):
        super().__init__(dialect, connection, checkfirst, tables)

    def visit_tiflash_replica(self, replica, **kwargs):
        """
        replica: TiFlashReplica
        """
        with self.with_ddl_events(replica):
            AlterTiFlashReplica(
                replica, new_num=replica.replica_num, **kwargs
            )._invoke_with(self.connection)

    def visit_vector_index(self, index, create_ok=False):
        """
        index: VectorIndex
        """
        if not create_ok and not self._can_create_index(index):
            return
        with self.with_ddl_events(index):
            # Automatically add tiflash replica if not exist
            if not index.table.info.get("has_tiflash_replica", False):
                num_replica = 1  # by default 1 replica
                replica = TiFlashReplica(index.table, num_replica)
                AlterTiFlashReplica(replica, new_num=num_replica)._invoke_with(
                    self.connection
                )
            # Create the vector index
            CreateVectorIndex(index)._invoke_with(self.connection)


class TiDBSchemaDropper(SchemaDropperBase):
    """Building logical DROP ... statements."""

    def __init__(self, dialect, connection, checkfirst=False, tables=None, **kwargs):
        super().__init__(dialect, connection, checkfirst, tables, **kwargs)

    def visit_tiflash_replica(self, replica, **kwargs):
        with self.with_ddl_events(replica):
            # set the replica_num to new_num
            AlterTiFlashReplica(replica, new_num=0)._invoke_with(self.connection)
            # reset the table.info of has_tiflash_replica
            del replica.inner_table.info["has_tiflash_replica"]

    def visit_vector_index(self, index, if_exists=False):
        # from IPython import embed;embed()
        if not self._can_drop_index(index):
            return
        with self.with_ddl_events(index):
            # Drop vector index is the same as dropping normal index
            sqlalchemy.sql.ddl.DropIndex(index, if_exists)._invoke_with(self.connection)
