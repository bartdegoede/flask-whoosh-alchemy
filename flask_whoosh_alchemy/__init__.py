from werkzeug.utils import import_string()

class WhooshAlchemy(object):
    def __init__(self, app=None, db=None):
        self.db = db
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        self._search = import_string('flask_whoosh_alchemy.search.Search')

    def __getattr__(self, name):
        return getattr(self._search, name)
