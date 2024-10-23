from typing import Dict, Any, Optional

from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import declarative_base

from ..ddl import Table, MetaData


class TiDBDeclarativeMeta(DeclarativeMeta):
    def __new__(cls, name: str, bases, class_dict: Dict[str, Any]):
        # Overwrite __table_cls__ to make sure all create table use
        # the custom `..ddl.Table` class.
        if "__table_cls__" not in class_dict:
            class_dict["__table_cls__"] = Table

        return DeclarativeMeta.__new__(cls, name, bases, class_dict)


def get_declarative_base(metadata: Optional[MetaData] = None):
    if metadata is None:
        metadata = MetaData()
    else:
        # ensure the metadata is created with type `..ddl.MetaData`
        assert isinstance(metadata, MetaData)
    return declarative_base(metadata=metadata, metaclass=TiDBDeclarativeMeta)
