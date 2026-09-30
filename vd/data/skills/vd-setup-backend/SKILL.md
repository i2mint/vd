---
name: vd-setup-backend
description: >-
  Install, start and verify a vector database for the vd package. Use this
  skill when the user needs to get a backend running — pip-installing a
  client, starting a server with Docker, setting cloud credentials (API keys,
  URIs, env vars), connecting vd to it, and smoke-testing the connection. Also
  trigger on BackendNotInstalledError, "connection refused" on a vector DB
  port, "how do I run qdrant / pgvector / redis / elasticsearch / milvus
  locally", or "vd.connect fails". For choosing which backend to use, see
  vd-backend-choose.
metadata:
  audience: users
---

# vd — install, start and verify a backend

The loop is always the same: **diagnose → act on the next step → re-diagnose →
connect → smoke-test.** `vd` carries the per-backend facts (pip extra, Docker
one-liner, env vars, docs links), so read them from `vd` instead of guessing.

## 1. Diagnose

```python
import vd

report = vd.check_requirements("qdrant")  # prints a report; returns a dict
report["ok"], report["next_step"]
```

It checks, depending on the backend's archetype:

- **embedded** (`memory`, `chroma`, `lancedb`, `sqlite_vec`, `duckdb`, `faiss`):
  the client library is importable, plus quirks (sqlite-vec needs SQLite ≥ 3.41
  and extension loading; Milvus Lite is not native-Windows).
- **server** (`qdrant`, `weaviate`, `milvus`, `redis`, `elasticsearch`,
  `pgvector`): the client is installed and something answers on the default
  port. For `qdrant` and `milvus`, which also run embedded, a missing server
  is reported but does not make the backend "not ready".
- **managed** (`pinecone`, `mongodb`, `turbopuffer`): the client is installed
  and the required environment variables are set.

`next_step` is always one concrete action: a pip command, a `docker run`
one-liner, or an `export VAR=...`. Do it, then call `check_requirements`
again until `report["ok"]` is `True`.

## 2. Get the full playbook

```python
print(vd.setup_guide("pgvector"))
```

prints the install command, the Docker one-liner (server backends), the
credentials to set (managed backends), a verify command and docs links.

## 3. Install the client

Every backend with a `vd` adapter installs through `vd`'s own extra, which
pins exactly the libraries the adapter imports:

```bash
pip install "vd[qdrant]"          # one backend (keep the quotes: zsh globs [])
pip install "vd[embedded]"        # chroma, qdrant, faiss, lancedb, sqlite_vec, duckdb
pip install "vd[all-backends]"    # every client
```

`vd.install_command("pgvector")` returns the command as a string, and
`vd.install_backend("pgvector", run=True)` runs it with the current
interpreter's pip. Ask the user before running installs on their machine.

## 4. Start the backend

**Embedded — nothing to start.** Pass a path to persist, omit it for a
throwaway store:

| Backend | Persist with | Default when omitted |
|---|---|---|
| `memory` | (never persists) | in-process dict |
| `chroma` | `persist_directory="./db"` | in-memory, shared by every chroma client in the process |
| `lancedb` | `path="./lance"` (or `s3://…`) | temp directory |
| `sqlite_vec` | `path="./vd.sqlite"` | `":memory:"` |
| `duckdb` | `path="./vd.duckdb"` | `":memory:"` |
| `faiss` | `path="./faiss_dir"` | in-memory |
| `qdrant` | `path="./qdrant_data"` | `":memory:"` |
| `milvus` | `path="./milvus.db"` (Milvus Lite) | temp `.db` |

**Server — run the container** from `vd.setup_guide(name)`, e.g.:

```bash
docker run -p 6379:6379 redis:8
docker run -p 9200:9200 -e discovery.type=single-node -e xpack.security.enabled=false docker.elastic.co/elasticsearch/elasticsearch:8.18.0
docker run -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg17
```

Wait until the port answers (Elasticsearch takes ~20 s). Disabling security,
as above, is for local development only.

**Managed — create the account and export credentials** (never commit them):

| Backend | Credentials | Read from the environment by vd? |
|---|---|---|
| `pinecone` | `PINECONE_API_KEY` | yes (`api_key=` overrides) |
| `mongodb` (Atlas) | `MONGODB_URI` | yes (`uri=` overrides) |
| `turbopuffer` | `TURBOPUFFER_API_KEY` | yes (`api_key=` overrides) |
| `pgvector` | `DATABASE_URL` or `POSTGRES_DSN` | yes (`dsn=` / `url=` override) |
| `qdrant` cloud | `QDRANT_URL`, `QDRANT_API_KEY` | no — pass `url=`, `api_key=` |
| `weaviate` cloud | `WEAVIATE_URL`, `WEAVIATE_API_KEY` | no — pass `url=`, `api_key=` |
| `milvus` / Zilliz | `MILVUS_URI`, `MILVUS_TOKEN` | no — pass `uri=`, `token=` |
| `elasticsearch` | `ELASTICSEARCH_URL`, `ELASTIC_API_KEY` | no — pass `url=`, `api_key=` |

**Managed, but no account for development:** two managed backends have
official local stand-ins, so you can build and test before signing up.

```bash
# Pinecone Local: in-memory emulator, any api_key works (use "pclocal").
docker run -d -p 5080-5090:5080-5090 -e PORT=5080 -e PINECONE_HOST=localhost ghcr.io/pinecone-io/pinecone-local
# MongoDB Atlas Local: real $vectorSearch, on host port 27018 here.
docker run -d -p 27018:27017 mongodb/mongodb-atlas-local
```

```python
vd.connect("pinecone", api_key="pclocal", host="http://localhost:5080")
vd.connect("mongodb", uri="mongodb://localhost:27018/?directConnection=true")
```

Pinecone Local speaks the pre-2026-07 API, so it needs `pip install "pinecone<10"`
for now. turbopuffer has no emulator: use a throwaway namespace on a real
account. If Docker Hub refuses anonymous pulls (rate limit), prefix Docker Hub
images with Google's mirror, e.g. `mirror.gcr.io/mongodb/mongodb-atlas-local`.

## 5. Connect

```python
vd.connect("redis", host="localhost", port=6379)  # or url="redis://..."
vd.connect("elasticsearch", url="http://localhost:9200")
vd.connect("pgvector", dsn="postgresql://user:pw@localhost:5432/db")
vd.connect("qdrant", url="http://localhost:6333")  # server
vd.connect("qdrant", url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
vd.connect("weaviate")  # localhost:8080 + gRPC 50051
vd.connect("milvus", uri="http://localhost:19530")  # server; path= for Lite
vd.connect("pinecone")  # PINECONE_API_KEY from env
vd.connect("mongodb")  # MONGODB_URI from env
```

Add `embedder=my_fn` only if you want to pass raw text (see **vd-quickstart**).

## 6. Smoke-test

Run this against the new client before building on it. It uses a unique
collection name and cleans up after itself:

```python
import uuid
import vd


def smoke_test(client, dim=3):
    """Create, write, read, search and drop a scratch collection."""
    name = f"vd_smoke_{uuid.uuid4().hex[:8]}"
    col = client.create_collection(name, dimension=dim)
    try:
        col["a"] = vd.Document(id="a", text="alpha", vector=[1.0, 0.0, 0.0])
        col["b"] = vd.Document(id="b", text="beta", vector=[0.0, 1.0, 0.0])
        assert col["a"].text == "alpha" and len(col) == 2
        top = next(iter(col.search([0.9, 0.1, 0.0], limit=1)))
        assert top["id"] == "a", top
    finally:
        client.delete_collection(name)
    return "ok"


smoke_test(vd.connect("memory"))
```

## Troubleshooting

- **`BackendNotInstalledError`** — the message contains the exact
  `pip install "vd[...]"` command.
- **Connection refused** — the server isn't up yet or is on another port;
  `check_requirements` probes the default port. Weaviate also needs gRPC port
  `50051` published.
- **sqlite-vec: "extension loading disabled"** — this Python's `sqlite3` was
  built without it (common with macOS system Python). Use a Homebrew/pyenv
  Python or `pysqlite3-binary`.
- **Milvus on Windows** — Milvus Lite has no native-Windows build; use WSL2
  or a Milvus server via `uri=`.
- **pgvector: an error about the `vector` extension** — the Postgres server
  lacks pgvector; use the `pgvector/pgvector` image or install the
  `postgresql-<ver>-pgvector` package on the server.
- **Free-tier limits and prices drift.** `vd.provider(name)["docs"]` links to
  the live pricing pages; re-check there rather than quoting numbers.

For contributors, the repo's `tests/docker-compose.yml` brings up pgvector,
Redis Stack, Elasticsearch, Weaviate, MongoDB Atlas Local, a Qdrant server and
Pinecone Local together for the live test suite, with no accounts
(`DOCKERHUB_MIRROR=mirror.gcr.io docker compose -f tests/docker-compose.yml up -d`).
