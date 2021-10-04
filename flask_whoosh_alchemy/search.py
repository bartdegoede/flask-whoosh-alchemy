import logging
import os
import sys

from flask_sqlalchemy import models_committed

import sqlalchemy
from sqlalchemy import types
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.hybrid import hybrid_property

from whoosh import index as whoosh_index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import BOOLEAN, DATETIME, ID, NUMERIC, TEXT
from whoosh.fields import Schema as _Schema
from whoosh.qparser import AndGroup, MultifieldParser, OrGroup

from werkzeug.utils import import_string

# if sys.version_info[0] < 3:
#     str = unicode

DEFAULT_ANALYZER = StemmingAnalyzer()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stderr))


def relation_column(instance, fields):
    '''
    such as: user.username
    such as: replies.content
    '''
    relation = getattr(instance.__class__, fields[0]).property
    _field = getattr(instance, fields[0])
    if relation.lazy == 'dynamic':
        _field = _field.first()
    return getattr(_field, fields[1]) if _field else ''


class Schema(object):
    def __init__(self, index):
        self.index = index
        self.pk = index.pk
        self.analyzer = index.analyzer
        self.schema = _Schema(**self.fields)

    def _fields(self):
        return {self.pk: ID(stored=True, unique=True)}

    def fields_map(self, field_type):
        if field_type == 'primary':
            return ID(stored=True, unique=True)
        type_map = {
            'date': types.Date,
            'datetime': types.DateTime,
            'boolean': types.Boolean,
            'integer': types.Integer,
            'float': types.Float
        }
        if isinstance(field_type, str):
            field_type = type_map.get(field_type, types.Text)

        if not isinstance(field_type, type):
            field_type = field_type.__class__

        if issubclass(field_type, (types.DateTime, types.Date)):
            return DATETIME(stored=True, sortable=True)
        elif issubclass(field_type, types.Integer):
            return NUMERIC(stored=True, numtype=int)
        elif issubclass(field_type, types.Float):
            return NUMERIC(stored=True, numtype=float)
        elif issubclass(field_type, types.Boolean):
            return BOOLEAN(stored=True)
        return TEXT(stored=True, analyzer=self.analyzer, sortable=False)

    @property
    def fields(self):
        model = self.index.model
        schema_fields = self._fields()
        primary_keys = [key.name for key in inspect(model).primary_key]

        schema = getattr(model, '__search_schema__', dict())

        for field in self.index.searchable:
            if '.' in field:
                fields = field.split('.')
                field_attr = getattr(getattr(model, fields[0]).property.mapper.class_, fields[1])
            else:
                field_attr = getattr(model, field)

            if field in schema:
                field_type = schema[field]
                if isinstance(field_type, str):
                    schema_fields[field] = self.fields_map(field_type)
                else:
                    schema_fields[field] = field_type
                continue

            if hasattr(field_attr, 'descriptor') and isinstance(field_attr.descriptor, hybrid_property):
                schema_fields[field] = self.fields_map('text')
                continue

            if field in primary_keys:
                schema_fields[field] = self.fields_map('text')
                continue

            field_type = field_attr.property.columns[0].type
            schema_fields[field] = self.fields_map(field_type)
        return schema_fields


class Index(object):
    def __init__(self, model, name, pk, analyzer, path=''):
        self.model = model
        self.path = path
        self.name = getattr(model, '__search_index__', name)
        self.pk = getattr(model, '__search_primary_key__', pk)
        self.analyzer = getattr(model, '__search_analyzer__', analyzer)
        self.searchable = set(getattr(model, '__searchable__', []))
        self._schema = Schema(self)
        self._writer = None
        self._client = self.init_reader()

    def init_reader(self):
        idx_path = os.path.join(self.path, self.name)
        if whoosh_index.exists_in(idx_path):
            return whoosh_index.open_dir(idx_path)
        if not os.path.exists(idx_path):
            os.makedirs(idx_path)
        return whoosh_index.create_in(idx_path, self.schema)

    @property
    def index(self):
        return self

    @property
    def fields(self):
        return self.schema.names()

    @property
    def schema(self):
        return self._schema.schema

    def create(self, *args, **kwargs):
        if self._writer is None:
            self._writer = self._client.writer()
        return self._writer.add_document(**kwargs)

    def update(self, *args, **kwargs):
        if self._writer is None:
            self._writer = self._client.writer()
        return self._writer.update_document(**kwargs)

    def delete(self, *args, **kwargs):
        if self._writer is None:
            self._writer = self._client.writer()
        return self._writer.delete_by_term(**kwargs)

    def commit(self):
        if self._writer is None:
            self._writer = self._client.writer()
        r = self._writer.commit()
        self._writer = None
        return r

    def search(self, *args, **kwargs):
        # TODO: memoize the searcher too
        return self._client.searcher().search(*args, **kwargs)


class Search(object):
    def __init__(self, app=None, db=None, analyzer=None):
        self._signal = None
        self._indices = dict()
        self.db = db
        self.analyzer = analyzer
        if app is not None:
            self.init_app(app)

    def _setdefault(self, app):
        app.config.setdefault('SEARCH_PRIMARY_KEY', 'id')
        app.config.setdefault('SEARCH_INDEX_NAME', 'search')
        app.config.setdefault('SEARCH_INDEX_SIGNAL', default_signal)
        app.config.setdefault('SEARCH_ANALYZER', None)
        app.config.setdefault('SEARCH_ENABLE', True)

    def _connect_signal(self, app):
        if app.config['SEARCH_ENABLE']:
            signal = app.config['SEARCH_INDEX_SIGNAL']
            if isinstance(signal, str):
                self._signal = import_string(signal)
            else:
                self._signal = signal
            models_committed.connect(self.index_signal)

    def index_signal(self, sender, changes):
        return self._signal(self, sender, changes)

    def init_app(self, app):
        self._setdefault(app)
        self._connect_signal(app)
        if self.analyzer is None:
            self.analyzer = app.config['SEARCH_ANALYZER'] or DEFAULT_ANALYZER
        self.pk = app.config['SEARCH_PRIMARY_KEY']
        self.index_name = app.config['SEARCH_INDEX_NAME']

        self.app = app
        if not self.db:
            self.db = self.app.extensions['sqlalchemy'].db
        self.db.Model.query_class = self._query_class(self.db.Model.query_class)

    def search(self, model, query, fields=None, limit=None, or_=True, **kwargs):
        index = self.index(model)
        if fields is None:
            fields = index.fields

        def _parser(fieldnames, schema, group, **kwargs):
            return MultifieldParser(fieldnames, schema, group=group, **kwargs)

        group = OrGroup if or_ else AndGroup
        parser = getattr(model, '__search_parser__', _parser)(
            fields,
            index.schema,
            group,
            **kwargs
        )

        return index.search(parser.parse(query), limit=limit)

    def _query_class(self, q):
        _self = self

        class Query(q):
            def search(self, query, fields=None, limit=None, or_=False, rank_order=False, **kwargs):
                model = self._mapper_zero().class_
                index = self.index(model)
                results = _self.search(model, query, fields=fields, limit=limit, or_=or_, **kwargs)

                if not results:
                    return self.filter(sqlalchemy.text('null'))
                result_set = {result[index.pk] for result in results}
                # build SQL query
                result_query = self.filter(getattr(model, index.pk).in_(result_set))
                if rank_order:
                    # order by relevance score
                    id_scores = {result[index.pk]: result.rank for result in results}
                    result_query.order_by(sqlalchemy.sql.case(id_scores, value=getattr(model, index.pk)))

                return result_query

        return Query

    def index(self, model):
        name = model.__table__.name
        if name not in self._indices:
            self._indices[name] = Index(model, name, self.pk, self.analyzer, self.index_name)
        return self._indices[name]

    def create_index(self, model='__all__', update=False, delete=False, yield_per=100):
        if model == '__all__':
            return self.create_all_indices(update=update, delete=delete)
        index = self.index(model)
        instances = model.query.enable_eager_loads(False).yield_per(yield_per)
        for instance in instances:
            self.create_one_index(instance, update=update, delete=delete, commit=False)
        index.commit()
        return index

    def create_all_indices(self, update=False, delete=False, yield_per=100):
        models = [model for model in self.db.Model._decl_class_registry.values() if hasttr(model, '__searchable__')]
        indices = []
        for model in models:
            indices.append(self.create_index(model, update, delete, yield_per))
        return indices

    def create_one_index(self, instance, update=False, delete=False, commit=True):
        if update and delete:
            raise ValueError("Can't update and delete in the same operation")

        index = self.index(instance.__class__)
        pk = index.pk
        attrs = {pk: str(getattr(instance, pk))}

        for field in index.fields:
            if '.' in field:
                attrs[field] = str(relation_column(instance, field.split('.')))
            else:
                attrs[field] = str(getattr(instance, field))
        if delete:
            logger.debug(f'Deleting index: {instance}')
            index.delete(fieldname=pk, text=str(getattr(instance, pk)))
        elif update:
            logger.debug(f'Updating index: {instance}')
            index.update(**attrs)
        else:
            logger.debug(f'Creating index: {instance}')
            index.create(**attrs)
        if commit:
            index.commit()
        return instance

    def update_one_index(self, instance, commit=True):
        return self.create_one_index(instance, update=True, commit=commit)

    def delete_one_index(self, instance, commit=True):
        return self.delete_one_index(instance, delete=True, commit=commit)

    def update_all_index(self, yield_per=100):
        return self.create_all_indices(update=True, yield_per=yield_per)

    def delete_all_index(self, yield_per=100):
        return self.create_all_indices(delete=True, yield_per=yield_per)

    def update_index(self, model='__all__', yield_per=100):
        return self.create_index(model, update=True, yield_per=yield_per)

    def delete_index(self, model='__all__', yield_per=100):
        return self.create_index(model, delete=True, yield_per=yield_per)
