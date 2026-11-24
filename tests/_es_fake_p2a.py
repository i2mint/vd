"""Fake elasticsearch module (sync+async) to exercise vd's native async ES adapter."""
import asyncio, sys, types

class ApiError(Exception): pass
class NotFoundError(ApiError): pass
class BadRequestError(ApiError): pass

class _Indices:
    def __init__(self, store): self.s = store
    async def exists(self, index):
        await asyncio.sleep(0); return index in self.s
    async def create(self, index, mappings=None, settings=None):
        await asyncio.sleep(0)
        if index in self.s:
            raise BadRequestError("resource_already_exists_exception")
        self.s[index] = {"mappings": mappings or {}, "docs": {}}
    async def get_mapping(self, index):
        await asyncio.sleep(0)
        if index not in self.s: raise NotFoundError(index)
        return {index: {"mappings": self.s[index]["mappings"]}}
    async def put_mapping(self, index, properties):
        await asyncio.sleep(0)
        self.s[index]["mappings"].setdefault("properties", {}).update(properties)
    async def refresh(self, index): await asyncio.sleep(0)
    async def delete(self, index):
        await asyncio.sleep(0)
        if index not in self.s: raise NotFoundError(index)
        del self.s[index]
    async def get(self, index):
        return {k: {} for k in self.s}

class AsyncElasticsearch:
    def __init__(self, hosts=None, **kw):
        self.store = {}; self.indices = _Indices(self.store); self.closed = False
    async def index(self, index, id, document):
        await asyncio.sleep(0)
        if index not in self.store:  # ES auto-creates on index
            self.store[index] = {"mappings": {}, "docs": {}}
        self.store[index]["docs"][id] = document
    async def get(self, index, id):
        await asyncio.sleep(0)
        try: return {"_source": self.store[index]["docs"][id]}
        except KeyError: raise NotFoundError(id)
    async def delete(self, index, id):
        await asyncio.sleep(0)
        if id not in self.store[index]["docs"]: raise NotFoundError(id)
        del self.store[index]["docs"][id]
    async def count(self, index):
        return {"count": len(self.store[index]["docs"])}
    async def close(self): self.closed = True

class Elasticsearch:  # unused, just needs to exist
    def __init__(self, *a, **k): pass

async def async_scan(es, index, **kw):
    for k in list(es.store[index]["docs"]):
        yield {"_id": k}

def scan(*a, **k): raise NotImplementedError

def install():
    m = types.ModuleType("elasticsearch")
    m.Elasticsearch = Elasticsearch; m.AsyncElasticsearch = AsyncElasticsearch
    m.NotFoundError = NotFoundError; m.BadRequestError = BadRequestError
    h = types.ModuleType("elasticsearch.helpers"); h.async_scan = async_scan; h.scan = scan
    m.helpers = h
    sys.modules["elasticsearch"] = m; sys.modules["elasticsearch.helpers"] = h
