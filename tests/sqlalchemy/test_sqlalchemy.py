import pytest
import numpy as np
import sqlalchemy
from sqlalchemy import URL, create_engine, Column, Integer, select
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError
from tidb_vector.sqlalchemy import VectorType
import tidb_vector.sqlalchemy as sqlalchemy_tidb
from ..config import TestConfig


db_url = URL(
    #"mysql+pymysql",
    "tidb+mysqldb",
    username=TestConfig.TIDB_USER,
    password=TestConfig.TIDB_PASSWORD,
    host=TestConfig.TIDB_HOST,
    port=TestConfig.TIDB_PORT,
    database="test",
    query={"ssl_verify_cert": True, "ssl_verify_identity": True}
    if TestConfig.TIDB_SSL
    else {},
)

engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Item1Model(Base):
    __tablename__ = "sqlalchemy_item1"
    id = Column(Integer, primary_key=True)
    embedding = Column(VectorType())


class Item2Model(Base):
    __tablename__ = "sqlalchemy_item2"
    id = Column(Integer, primary_key=True)
    embedding = Column(VectorType(dim=3))


class TestSQLAlchemy:
    def setup_class(self):
        Item1Model.__table__.drop(bind=engine, checkfirst=True)
        Item1Model.__table__.create(bind=engine)

    def teardown_class(self):
        Item1Model.__table__.drop(bind=engine, checkfirst=True)

    def setup_method(self):
        with Session() as session:
            session.query(Item1Model).delete()
            session.commit()

    def test_insert_get_record(self):
        with Session() as session:
            item1 = Item1Model(embedding=[1, 2, 3])
            session.add(item1)
            session.commit()
            item1 = session.query(Item1Model).first()
            assert np.array_equal(item1.embedding, np.array([1, 2, 3]))
            assert item1.embedding.dtype == np.float32

    def test_empty_vector(self):
        with Session() as session:
            item1 = Item1Model(embedding=[])
            session.add(item1)
            session.commit()
            assert session.query(Item1Model).count() == 1
            item1 = session.query(Item1Model).first()
            assert np.array_equal(item1.embedding, np.array([]))
            assert item1.embedding.dtype == np.float32

    def test_get_with_different_dimensions(self):
        with Session() as session:
            item1 = Item1Model(embedding=[1, 2, 3])
            session.add(item1)
            session.commit()

            with pytest.raises(OperationalError) as excinfo:
                session.scalar(
                    select(Item1Model.embedding.l1_distance([1, 2, 3, 4]) < 0.1)
                )
            assert "vectors have different dimensions" in str(excinfo.value)

    def test_l1_distance(self):
        with Session() as session:
            item1 = Item1Model(embedding=[1, 2, 3])
            item2 = Item1Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            result = session.scalars(
                select(Item1Model).filter(
                    Item1Model.embedding.l1_distance([1, 2, 3.1]) < 0.2
                )
            ).all()
            assert len(result) == 2

            distance = Item1Model.embedding.l1_distance([1, 2, 3])
            items = (
                session.query(Item1Model.id, distance.label("distance"))
                .order_by(distance)
                .limit(5)
                .all()
            )
            assert len(items) == 2

    def test_l2_distance(self):
        with Session() as session:
            item1 = Item1Model(embedding=[1, 2, 3])
            item2 = Item1Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            result = session.scalars(
                select(Item1Model).filter(
                    Item1Model.embedding.l2_distance([1, 2, 3.1]) < 0.2
                )
            ).all()
            assert len(result) == 2

            distance = Item1Model.embedding.l2_distance([1, 2, 3])
            items = (
                session.query(Item1Model.id, distance.label("distance"))
                .order_by(distance)
                .limit(5)
                .all()
            )
            assert len(items) == 2
            assert items[0].distance == 0.0

    def test_cosine_distance(self):
        with Session() as session:
            item1 = Item1Model(embedding=[1, 2, 3])
            item2 = Item1Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            result = session.scalars(
                select(Item1Model).filter(
                    Item1Model.embedding.cosine_distance([1, 2, 3.1]) < 0.2
                )
            ).all()
            assert len(result) == 2

            distance = Item1Model.embedding.cosine_distance([1, 2, 3])
            items = (
                session.query(Item1Model.id, distance.label("distance"))
                .order_by(distance)
                .limit(5)
                .all()
            )
            assert len(items) == 2

    def test_negative_inner_product(self):
        with Session() as session:
            item1 = Item1Model(embedding=[1, 2, 3])
            item2 = Item1Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            result = session.scalars(
                select(Item1Model).filter(
                    Item1Model.embedding.negative_inner_product([1, 2, 3.1]) < -14.0
                )
            ).all()
            assert len(result) == 2

            distance = Item1Model.embedding.negative_inner_product([1, 2, 3])
            items = (
                session.query(Item1Model.id, distance.label("distance"))
                .order_by(distance)
                .limit(5)
                .all()
            )
            assert len(items) == 2
            assert items[1].distance == -14.0


class TestSQLAlchemyWithDifferentDimensions:
    def setup_class(self):
        Item2Model.__table__.drop(bind=engine, checkfirst=True)
        Item2Model.__table__.create(bind=engine)

    def teardown_class(self):
        Item2Model.__table__.drop(bind=engine, checkfirst=True)

    def setup_method(self):
        with Session() as session:
            session.query(Item2Model).delete()
            session.commit()

    def test_insert_get_record(self):
        with Session() as session:
            item = Item2Model(embedding=[1, 2, 3])
            session.add(item)
            session.commit()
            item = session.query(Item2Model).first()
            assert np.array_equal(item.embedding, np.array([1, 2, 3]))
            assert item.embedding.dtype == np.float32

    def test_get_with_different_dimensions(self):
        with Session() as session:
            item = Item2Model(embedding=[1, 2, 3])
            session.add(item)
            session.commit()

            with pytest.raises(OperationalError) as excinfo:
                session.scalar(
                    select(Item2Model.embedding.l1_distance([1, 2, 3, 4]) < 0.1)
                )
            assert "vectors have different dimensions" in str(excinfo.value)

    def test_l1_distance(self):
        with Session() as session:
            item1 = Item2Model(embedding=[1, 2, 3])
            item2 = Item2Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            result = session.scalars(
                select(Item2Model).filter(
                    Item2Model.embedding.l1_distance([1, 2, 3.1]) < 0.2
                )
            ).all()
            assert len(result) == 2

            distance = Item2Model.embedding.l1_distance([1, 2, 3])
            items = (
                session.query(Item2Model.id, distance.label("distance"))
                .order_by(distance)
                .limit(5)
                .all()
            )
            assert len(items) == 2

    def test_l2_distance(self):
        with Session() as session:
            item1 = Item2Model(embedding=[1, 2, 3])
            item2 = Item2Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            result = session.scalars(
                select(Item2Model).filter(
                    Item2Model.embedding.l2_distance([1, 2, 3.1]) < 0.2
                )
            ).all()
            assert len(result) == 2

            distance = Item2Model.embedding.l2_distance([1, 2, 3])
            items = (
                session.query(Item2Model.id, distance.label("distance"))
                .order_by(distance)
                .limit(5)
                .all()
            )
            assert len(items) == 2
            assert items[0].distance == 0.0

    def test_cosine_distance(self):
        with Session() as session:
            item1 = Item2Model(embedding=[1, 2, 3])
            item2 = Item2Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            result = session.scalars(
                select(Item2Model).filter(
                    Item2Model.embedding.cosine_distance([1, 2, 3.1]) < 0.2
                )
            ).all()
            assert len(result) == 2

            distance = Item2Model.embedding.cosine_distance([1, 2, 3])
            items = (
                session.query(Item2Model.id, distance.label("distance"))
                .order_by(distance)
                .limit(5)
                .all()
            )
            assert len(items) == 2
            assert items[0].distance == 0.0

    def test_negative_inner_product(self):
        with Session() as session:
            item1 = Item2Model(embedding=[1, 2, 3])
            item2 = Item2Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            result = session.scalars(
                select(Item2Model).filter(
                    Item2Model.embedding.negative_inner_product([1, 2, 3.1]) < -14.0
                )
            ).all()
            assert len(result) == 2

            distance = Item2Model.embedding.negative_inner_product([1, 2, 3])
            items = (
                session.query(Item2Model.id, distance.label("distance"))
                .order_by(distance)
                .limit(5)
                .all()
            )
            assert len(items) == 2
            assert items[1].distance == -14.0

class TestSQLAlchemyDDL:
    def setup_class(self):
        Item2Model.__table__.drop(bind=engine, checkfirst=True)
        Item2Model.__table__.create(bind=engine)

    def teardown_class(self):
        Item2Model.__table__.drop(bind=engine, checkfirst=True)

    def setup_method(self):
        with Session() as session:
            session.query(Item2Model).delete()
            session.commit()

    def test_l2_distance(self):
        with Session() as session:
            # indexes
            l2_index = sqlalchemy_tidb.ddl.VectorIndex(
                'idx_embedding_l2',
                sqlalchemy.func.vec_l2_distance(Item2Model.__table__.c.embedding),
            )
            l2_index.create(engine)
            cos_index = sqlalchemy_tidb.ddl.VectorIndex(
                'idx_embedding_cos',
                sqlalchemy.func.vec_cosine_distance(Item2Model.__table__.c.embedding),
            )
            cos_index.create(engine)

            item1 = Item2Model(embedding=[1, 2, 3])
            item2 = Item2Model(embedding=[1, 2, 3.2])
            session.add_all([item1, item2])
            session.commit()

            # l2 distance
            result_l2 = session.scalars(
                select(Item2Model).filter(
                    Item2Model.embedding.l2_distance([1, 2, 3.1]) < 0.2
                )
            ).all()
            assert len(result_l2) == 2

            distance_l2 = Item2Model.embedding.l2_distance([1, 2, 3])
            items_l2 = (
                session.query(Item2Model.id, distance_l2.label("distance"))
                .order_by(distance_l2)
                .limit(5)
                .all()
            )
            assert len(items_l2) == 2
            assert items_l2[0].distance == 0.0

            # cosine distance
            result_cos = session.scalars(
                select(Item2Model).filter(
                    Item2Model.embedding.cosine_distance([1, 2, 3.1]) < 0.2
                )
            ).all()
            assert len(result_cos) == 2

            distance_cos = Item2Model.embedding.cosine_distance([1, 2, 3])
            items_cos = (
                session.query(Item2Model.id, distance_cos.label("distance"))
                .order_by(distance_cos)
                .limit(5)
                .all()
            )
            assert len(items_cos) == 2
            assert items_cos[0].distance == 0.0
