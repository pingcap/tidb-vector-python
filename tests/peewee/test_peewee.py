import numpy as np
import pytest
from peewee import MySQLDatabase, Model, OperationalError
from tidb_vector.peewee import VectorField
from ..config import TestConfig

try:
    import pymysql  # noqa

    connect_kwargs = {"ssl_verify_cert": True, "ssl_verify_identity": True}
except ImportError:
    connect_kwargs = {}


db = MySQLDatabase(
    "ci_peewee_test",
    host=TestConfig.TIDB_HOST,
    port=TestConfig.TIDB_PORT,
    user=TestConfig.TIDB_USER,
    password=TestConfig.TIDB_PASSWORD,
    **connect_kwargs
)


class BaseModel(Model):
    class Meta:
        database = db


class Item1Model(BaseModel):
    embedding = VectorField()


class Item2Model(BaseModel):
    embedding = VectorField(dimensions=3)


class TestPeewee:
    def setup_class(self):
        db.connect()
        db.drop_tables([Item1Model])
        db.create_tables([Item1Model])

    def teardown_class(self):
        db.drop_tables([Item1Model])
        db.close()

    def setup_method(self):
        Item1Model.truncate_table()

    def test_insert_get_record(self):
        Item1Model.create(embedding=[1, 2, 3])
        assert Item1Model.select().count() == 1
        item1 = Item1Model.get()
        assert np.array_equal(item1.embedding, np.array([1, 2, 3]))
        assert item1.embedding.dtype == np.float32

    def test_get_with_different_dimensions(self):
        Item1Model.create(embedding=[1, 2, 3])
        with pytest.raises(OperationalError) as excinfo:
            items = Item1Model.select().where(
                Item1Model.embedding.l1_distance([1, 2, 3, 4]) < 0.1
            )
            print(items.count())
        assert "vectors have different dimensions" in str(excinfo.value)

    def test_l1_distance(self):
        Item1Model.create(embedding=[1, 2, 3])
        item1 = Item1Model.get()
        query = Item1Model.select().where(
            Item1Model.embedding.l1_distance([1, 2, 3.1]) < 0.1
        )
        assert query.count() == 1
        assert query.get().id == item1.id

        distance = Item1Model.embedding.l1_distance([1, 2, 3])
        items = (
            Item1Model.select(Item1Model.id, distance.alias("distance"))
            .order_by(distance)
            .limit(5)
        )
        assert items.count() == 1
        assert items.get().id == item1.id
        assert items[0].distance == 0.0

    def test_l2_distance(self):
        Item1Model.create(embedding=[1, 2, 3])
        item1 = Item1Model.get()
        query = Item1Model.select().where(
            Item1Model.embedding.l2_distance([1, 2, 3.1]) < 0.1
        )
        assert query.count() == 1
        assert query.get().id == item1.id

        distance = Item1Model.embedding.l2_distance([1, 2, 3])
        items = (
            Item1Model.select(Item1Model.id, distance.alias("distance"))
            .order_by(distance)
            .limit(5)
        )
        assert items.count() == 1
        assert items.get().id == item1.id
        assert items[0].distance == 0.0

    def test_cosine_distance(self):
        Item1Model.create(embedding=[1, 2, 3])
        item1 = Item1Model.get()
        query = Item1Model.select().where(
            Item1Model.embedding.cosine_distance([1, 2, 3.1]) < 0.1
        )
        assert query.count() == 1
        assert query.get().id == item1.id

        distance = Item1Model.embedding.cosine_distance([1, 2, 3])
        items = (
            Item1Model.select(Item1Model.id, distance.alias("distance"))
            .order_by(distance)
            .limit(5)
        )
        assert items.count() == 1
        assert items.get().id == item1.id
        assert items[0].distance == 0.0

    def test_negative_inner_product(self):
        Item1Model.create(embedding=[1, 2, 3])
        item1 = Item1Model.get()
        """
        tidb> select vec_negative_inner_product('[1,2,3]', '[1,2,3.1]');
        +----------------------------------------------------+
        | vec_negative_inner_product('[1,2,3]', '[1,2,3.1]') |
        +----------------------------------------------------+
        |                                -14.299999237060547 |
        +----------------------------------------------------+
        """
        query = Item1Model.select().where(
            Item1Model.embedding.negative_inner_product([1, 2, 3.1]) < -14
        )
        assert query.count() == 1
        assert query.get().id == item1.id

        distance = Item1Model.embedding.negative_inner_product([1, 2, 3])
        items = (
            Item1Model.select(Item1Model.id, distance.alias("distance"))
            .order_by(distance)
            .limit(5)
        )
        assert items.count() == 1
        assert items.get().id == item1.id
        assert items[0].distance == -14


class TestPeeweeWithExplicitDimensions:
    def setup_class(self):
        db.connect()
        db.drop_tables([Item2Model])
        db.create_tables([Item2Model])

    def teardown_class(self):
        db.drop_tables([Item2Model])
        db.close()

    def setup_method(self):
        Item2Model.truncate_table()

    def test_insert_get_record(self):
        Item2Model.create(embedding=[1, 2, 3])
        assert Item2Model.select().count() == 1
        item2 = Item2Model.get()
        assert np.array_equal(item2.embedding, np.array([1, 2, 3]))
        assert item2.embedding.dtype == np.float32

    def test_get_with_different_dimensions(self):
        Item2Model.create(embedding=[1, 2, 3])
        with pytest.raises(OperationalError) as excinfo:
            items = Item2Model.select().where(
                Item2Model.embedding.l1_distance([1, 2, 3, 4]) < 0.1
            )
            print(items.count())
        assert "vectors have different dimensions" in str(excinfo.value)

    def test_l1_distance(self):
        Item2Model.create(embedding=[1, 2, 3])
        item = Item2Model.get()
        query = Item2Model.select().where(
            Item2Model.embedding.l1_distance([1, 2, 3.1]) < 0.1
        )
        assert query.count() == 1
        assert query.get().id == item.id

        distance = Item2Model.embedding.l1_distance([1, 2, 3])
        items = (
            Item2Model.select(Item2Model.id, distance.alias("distance"))
            .order_by(distance)
            .limit(5)
        )
        assert items.count() == 1
        assert items.get().id == item.id
        assert items[0].distance == 0.0

    def test_l2_distance(self):
        Item2Model.create(embedding=[1, 2, 3])
        item = Item2Model.get()
        query = Item2Model.select().where(
            Item2Model.embedding.l2_distance([1, 2, 3.1]) < 0.1
        )
        assert query.count() == 1
        assert query.get().id == item.id

        distance = Item2Model.embedding.l2_distance([1, 2, 3])
        items = (
            Item2Model.select(Item2Model.id, distance.alias("distance"))
            .order_by(distance)
            .limit(5)
        )
        assert items.count() == 1
        assert items.get().id == item.id
        assert items[0].distance == 0.0

    def test_cosine_distance(self):
        Item2Model.create(embedding=[1, 2, 3])
        item = Item2Model.get()
        query = Item2Model.select().where(
            Item2Model.embedding.cosine_distance([1, 2, 3.1]) < 0.1
        )
        assert query.count() == 1
        assert query.get().id == item.id

        distance = Item2Model.embedding.cosine_distance([1, 2, 3])
        items = (
            Item2Model.select(Item2Model.id, distance.alias("distance"))
            .order_by(distance)
            .limit(5)
        )
        assert items.count() == 1
        assert items.get().id == item.id
        assert items[0].distance == 0.0

    def test_negative_inner_product(self):
        Item2Model.create(embedding=[1, 2, 3])
        item = Item2Model.get()
        """
        tidb> select vec_negative_inner_product('[1,2,3]', '[1,2,3.1]');
        +----------------------------------------------------+
        | vec_negative_inner_product('[1,2,3]', '[1,2,3.1]') |
        +----------------------------------------------------+
        |                                -14.299999237060547 |
        +----------------------------------------------------+
        """
        query = Item2Model.select().where(
            Item2Model.embedding.negative_inner_product([1, 2, 3.1]) < -14
        )
        assert query.count() == 1
        assert query.get().id == item.id

        distance = Item2Model.embedding.negative_inner_product([1, 2, 3])
        items = (
            Item2Model.select(Item2Model.id, distance.alias("distance"))
            .order_by(distance)
            .limit(5)
        )
        assert items.count() == 1
        assert items.get().id == item.id
        assert items[0].distance == -14
