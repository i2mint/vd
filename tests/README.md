# vd test suite

The headline file is `test_core.py` — the **facade contract suite**. Every test
there takes the `client` fixture, which is parametrized over every backend, so a
single test asserts the same behaviour across all of them. That is how `vd`
proves one uniform interface over many vector databases.

## Backends the suite sweeps over

`conftest.py` defines two groups:

- **Embedded backends** — `memory`, `chroma`, `faiss`, `duckdb`, `lancedb`,
  `qdrant`, `sqlite_vec`, `milvus`. No server needed; each test gets a fresh
  client. `sqlite_vec` is skipped on a Python whose `sqlite3` lacks
  loadable-extension support; `milvus` runs against the embedded **Milvus Lite**
  engine and is skipped if `milvus-lite` is not installed.
- **Server backends** — `pgvector`, `redis`, `elasticsearch`, `weaviate`,
  `mongodb`. Each needs a running container. A backend whose port is not open is
  **skipped**, so the suite stays green in a plain CI environment.

## Running against the server backends

```bash
# Start every server backend (first run pulls images — a few minutes).
docker compose -f tests/docker-compose.yml up -d

# Wait until all five report (healthy), then run the suite.
docker compose -f tests/docker-compose.yml ps

pytest tests/

# Tear down and wipe volumes when done.
docker compose -f tests/docker-compose.yml down -v
```

Run one backend at a time with `-k`:

```bash
pytest tests/test_core.py -k pgvector
```

## Connection settings

The server backends use these defaults; override with environment variables:

| Backend         | Env var               | Default                                              |
|-----------------|-----------------------|------------------------------------------------------|
| `pgvector`      | `VD_PGVECTOR_DSN`     | `postgresql://vd:vd@localhost:5432/vd`               |
| `redis`         | `VD_REDIS_HOST` / `VD_REDIS_PORT` | `localhost` / `6379`                     |
| `elasticsearch` | `VD_ELASTICSEARCH_URL`| `http://localhost:9200`                              |
| `weaviate`      | `VD_WEAVIATE_HOST`    | `localhost`                                          |
| `mongodb`       | `VD_MONGODB_URI`      | `mongodb://localhost:27018/?directConnection=true`   |

`mongodb` maps to host port **27018** (not 27017) to avoid colliding with a
developer's native `mongod`. It uses the `mongodb-atlas-local` image because
`$vectorSearch` requires an Atlas-capable deployment.

`milvus` needs no container — it is verified against the embedded Milvus Lite
engine, which exercises the same adapter code path as a Milvus server (only the
client constructor differs).
