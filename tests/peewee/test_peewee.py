import numpy as np
import pytest
from peewee import MySQLDatabase, Model, OperationalError
import tidb_vector
from tidb_vector.peewee import VectorField, VectorAdaptor
from ..config import TestConfig


if TestConfig.TIDB_SSL:
    connect_kwargs = {"ssl_verify_cert": True, "ssl_verify_identity": True}
else:
    connect_kwargs = {}


db = MySQLDatabase(
    "test",
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

    class Meta:
        table_name = "peewee_item1"


class Item2Model(BaseModel):
    embedding = VectorField(dimensions=3)

    class Meta:
        table_name = "peewee_item2"


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

    def test_empty_vector(self):
        Item1Model.create(embedding=[])
        assert Item1Model.select().count() == 1
        item1 = Item1Model.get()
        assert np.array_equal(item1.embedding, np.array([]))
        assert item1.embedding.dtype == np.float32

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


class TestPeeweeAdaptor:
    def setup_method(self):
        db.drop_tables([Item1Model, Item2Model])
        db.create_tables([Item1Model, Item2Model])

    def teardown_method(self):
        db.drop_tables([Item1Model, Item2Model])

    def test_create_index_on_dyn_vector(self):
        adaptor = VectorAdaptor(db)
        with pytest.raises(ValueError):
            adaptor.create_vector_index(
                Item1Model.embedding, distance_metric=tidb_vector.DistanceMetric.L2
            )
        assert adaptor.has_vector_index(Item1Model.embedding) is False

    def test_create_index_on_fixed_vector(self):
        adaptor = VectorAdaptor(db)
        adaptor.create_vector_index(
            Item2Model.embedding, distance_metric=tidb_vector.DistanceMetric.L2
        )
        assert adaptor.has_vector_index(Item2Model.embedding) is True

        with pytest.raises(Exception):
            adaptor.create_vector_index(
                Item2Model.embedding, distance_metric=tidb_vector.DistanceMetric.L2
            )

        assert adaptor.has_vector_index(Item2Model.embedding) is True

        adaptor.create_vector_index(
            Item2Model.embedding,
            distance_metric=tidb_vector.DistanceMetric.L2,
            skip_existing=True,
        )

        adaptor.create_vector_index(
            Item2Model.embedding,
            distance_metric=tidb_vector.DistanceMetric.COSINE,
            skip_existing=True,
        )

    def test_index_and_search(self):
        adaptor = VectorAdaptor(db)
        adaptor.create_vector_index(
            Item2Model.embedding, distance_metric=tidb_vector.DistanceMetric.L2
        )
        assert adaptor.has_vector_index(Item2Model.embedding) is True

        Item2Model.insert_many(
            [
                {"embedding": [1, 2, 3]},
                {"embedding": [1, 2, 3.2]},
            ]
        ).execute()

        distance = Item2Model.embedding.cosine_distance([1, 2, 3])
        items = (
            Item2Model.select(distance.alias("distance")).order_by(distance).limit(5)
        )
        assert items.count() == 2
        assert items[0].distance == 0.0
