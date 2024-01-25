import uuid

import sqlalchemy
from sqlalchemy.orm import declarative_base, DeclarativeMeta
from tidb_vector.sqlalchemy import VectorType

Base: DeclarativeMeta = declarative_base()


def get_simpledoc_table_model(table_name: str):
    """Create a simpledoc table class."""

    class SimpleDocsTableModel(Base):
        """
        This class represents a table for storing simple document data within a database.
        It is designed to hold various types of documents, with each document having associated metadata
        that can be used for filtering during searches.

        Attributes:
            id: A unique identifier for each document. It is generated automatically using UUID4,
                ensuring that each document has a distinct ID.
            content: The content (chunk) stored as text. This field is designed to hold
                the body of the document, which can be in any text format.
            content_id: A reference to the (chunk) content. This field stores its unique ID,
                allowing for a direct link between the vector and its corresponding table.
                It also used to delete and query specific documents.
            meta: A JSON field to store additional metadata about the document.
                Different from the meta of vector table, it will not be used for vector search.
        """

        __tablename__ = table_name
        id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
        content_id = sqlalchemy.Column(
            sqlalchemy.String(36),
            default=lambda: str(uuid.uuid4()),
            unique=True,
            nullable=False,
        )

        content = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
        meta = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)
        create_time = sqlalchemy.Column(
            sqlalchemy.DateTime, server_default=sqlalchemy.text("CURRENT_TIMESTAMP")
        )
        update_time = sqlalchemy.Column(
            sqlalchemy.DateTime,
            server_default=sqlalchemy.text(
                "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            ),
        )

        @classmethod
        def insert(cls, session, content, meta, commit_transaction=True):
            """
            Insert a new simple document into the database.

            :param session: The SQLAlchemy session to use for the operation.
            :param content: The content of the document.
            :param meta: The metadata associated with the document.
            :param commit_transaction: If True, commit the transaction after the insert. Otherwise, only flush the session.
            :return: The ID of the newly inserted document.
            """
            new_doc = cls(content=content, meta=meta)
            session.add(new_doc)
            if commit_transaction:
                session.commit()
            else:
                session.flush()  # Ensure the ID is assigned without committing the transaction
            return new_doc.content_id

        @classmethod
        def delete(cls, session, content_ids, commit_transaction=True):
            """
            Delete simple documents by their IDs in a batch.

            :param session: The SQLAlchemy session to use for the operation.
            :param content_ids: A list of content IDs of the documents to delete.
            :param commit_transaction: If True, commit the transaction after the delete. Otherwise, only flush the session.
            """
            if not isinstance(content_ids, list):
                content_ids = [
                    content_ids
                ]  # Ensure content_ids is a list even if a single ID is provided
            session.query(cls).filter(cls.content_id.in_(content_ids)).delete(
                synchronize_session="fetch"
            )
            if commit_transaction:
                session.commit()
            else:
                session.flush()  # Apply changes without committing the transaction

        @classmethod
        def get_by_ids(cls, session, content_ids):
            """
            Retrieve simple documents by their content IDs in a batch.

            :param session: The SQLAlchemy session to use for the operation.
            :param content_ids: A list of content IDs of the documents to retrieve.
            :return: A list of SimpleDocsTableModel instances corresponding to the given IDs.
            """
            if not isinstance(content_ids, list):
                content_ids = [
                    content_ids
                ]  # Ensure content_ids is a list even if a single ID is provided
            return session.query(cls).filter(cls.content_id.in_(content_ids)).all()

    return SimpleDocsTableModel


def get_vector_table_model(table_name: str):
    class VectorTableModel(Base):
        """
        This class represents a table designed to store vector data associated with documents.
        Each vector entry is linked to a document, enabling features like document similarity
        search and other vector-based operations.

        Attributes:
            id: A unique identifier for each vector entry, generated automatically using UUID4.
            embedding(VectorType): The vector data associated with a content.
            content_id: A reference to the (chunk) content this vector is associated with. This field
                stores its unique ID, allowing for a direct link between the vector and its corresponding table.
            ref_content_table: A string field specifying the name of the table this vector is associated with.
                This allows the system to differentiate between vectors associated with various types of
                documents or content that might be stored in separate tables.
            meta: A JSON field to store additional metadata about the document. This can include properties
                like the document's title, custom document IDs, and any other searchable data that might be useful
                for filtering or categorization purposes.
        """

        __tablename__ = table_name
        id = sqlalchemy.Column(
            sqlalchemy.String(36), primary_key=True, default=lambda: str(uuid.uuid4())
        )
        embedding = sqlalchemy.Column(VectorType())
        content_id = sqlalchemy.Column(
            sqlalchemy.String(36), nullable=False, index=True
        )
        ref_content_table = sqlalchemy.Column(
            sqlalchemy.String(255), nullable=False, index=True
        )
        meta = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)
        create_time = sqlalchemy.Column(
            sqlalchemy.DateTime, server_default=sqlalchemy.text("CURRENT_TIMESTAMP")
        )
        update_time = sqlalchemy.Column(
            sqlalchemy.DateTime,
            server_default=sqlalchemy.text(
                "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            ),
        )

    return VectorTableModel
