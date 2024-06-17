################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
################################

from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine
import sqlglot


def get_schema(db):
    base = automap_base()

    # engine
    engine = create_engine(f'sqlite:///{db}')

    # reflect the tables
    base.prepare(autoload_with=engine)
    return base


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema):
        self._schema = schema
        self._id_map = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def id_map(self):
        return self._id_map

    @property
    def tables(self) -> list[str]:
        return [table.__table__.name.lower() for table in self._schema.classes]

    @property
    def columns(self) -> dict[str, list[str]]:
        columns = {}
        for table in self._schema.classes:
            columns[table.__table__.name.lower()] = [col.name.lower() for col in table.__table__.columns]
        return columns

    def _map(self, schema):
        id_map = {'*': "__all__"}
        idx = 1
        for relation in schema.classes:
            for col in relation.__table__.columns:
                id_map[
                    relation.__table__.name.lower() + "." + col.name.lower()] = "__" + relation.__table__.name.lower() + "." + col.name.lower() + "__"
                idx += 1
            id_map[relation.__table__.name.lower()] = "__" + relation.__table__.name.lower() + "__"

        return id_map

    def to_sqlglot(self):
        tables = {}
        for table in self._schema.classes:
            col_dict = {
            }
            for col in table.__table__.columns:
                col_dict[col.name] = col.type
            tables[table.__table__.name] = col_dict
        return tables


def get_sql(schema, query):
    parsed = sqlglot.parse(query, dialect='sqlite')
    assert len(parsed) == 1, "Only support single query"
    parsed = parsed[0]
    return parsed
