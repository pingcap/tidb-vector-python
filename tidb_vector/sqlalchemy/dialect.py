import sqlalchemy.dialects.mysql as sqlalchemy_mysql

from .compiler import TiDBDDLCompiler


class TiDBDialect_mysqldb(sqlalchemy_mysql.mysqldb.MySQLDialect_mysqldb):
    name = "tidb"
    driver = "mysqldb"

    preparer = sqlalchemy_mysql.base.MySQLIdentifierPreparer
    ddl_compiler = TiDBDDLCompiler
    statement_compiler = sqlalchemy_mysql.base.MySQLCompiler

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, connection):
        super().initialize(connection)

    @classmethod
    def import_dbapi(cls):
        return __import__("MySQLdb")


dialect = TiDBDialect_mysqldb
