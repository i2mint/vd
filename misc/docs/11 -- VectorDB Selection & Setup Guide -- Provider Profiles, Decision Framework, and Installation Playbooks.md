# VectorDB Selection & Setup Guide — Provider Profiles, a Decision Framework, and Installation Playbooks for Practitioners (2026)

**File:** `11 -- VectorDB Selection & Setup Guide -- Provider Profiles, Decision Framework, and Installation Playbooks.md`
**Author:** Thor Whalen
**As-of:** May 2026 — this is a practitioner snapshot. Pricing, free-tier limits, package names, and default indexes drift on the order of months; treat any number here as a pointer to the live docs/pricing pages referenced, not a contract. Companion to Report 03 (theory: ANN indexes, facade contract, filter dialects, hybrid abstraction) [1], Report 02 (embedding models), and Report 08 (browser/WebGPU search, out of scope here).

## TL;DR

- **For most practitioners in 2026, three backends cover ≥90% of situations:** **Chroma** for embedded prototypes and small persistent apps, **Qdrant** for self-host-or-managed with a clean Python API (free 1 GB cluster forever) [6], and **pgvector 0.8.2** if you already run Postgres [30]. Reach for **Pinecone** [4] or **Turbopuffer** [42] when you want zero ops and serverless billing; reach for **Milvus/Zilliz** [16] or **Vespa** [47] only at billion-scale.
- **Watch four 2024–2026 changes that invalidate older guides:** (1) the Pinecone Python package was renamed `pinecone-client` → `pinecone` at v5.1.0 (current v9.0.0, requires Python ≥3.10) [1][2]; (2) Weaviate Cloud restructured pricing in October 2025 — old "Serverless $25/mo" is gone, replaced by Flex ($45/mo minimum), Plus ($280/mo, annual), Premium ($400/mo), and the 14-day sandbox is no longer extendable [10][11]; (3) Redis 8 returned to open source under AGPLv3 in May 2025 and rolled RediSearch/RedisJSON into core [55]; Elasticsearch added AGPLv3 as a third option in August 2024 [56][39]; (4) `pgvecto.rs` is deprecated in favour of **VectorChord** (`vchord`) [49][50] and Marqo's OSS project is officially deprecated on its GitHub README [52].
- **Treat embedding as external and optional in your facade.** Some backends embed text for you (Chroma's default Sentence Transformers [19], Weaviate vectorizer modules [12], Pinecone integrated inference [3], MongoDB Atlas Auto Embeddings [41], Vertex AI Vector Search 2.0 [53], Azure AI Search integrated vectorization [54]); most do not. A facade that hard-codes "the DB embeds for me" will not survive contact with Qdrant, Milvus, FAISS, sqlite-vec, DuckDB-VSS, LanceDB, pgvector, Turbopuffer, VectorChord, or Vald.

## Key Findings

1. **The deployment archetype dominates setup effort, not the algorithm.** Embedded (pip-only) backends — Chroma local [19], LanceDB [21], sqlite-vec [24], DuckDB-VSS [28], FAISS [45] — are running in 60 seconds. Self-hosted server backends — Qdrant [9], Weaviate, Milvus [16], Redis 8, Elasticsearch — need a Docker daemon and 2–8 GB RAM. Managed (Pinecone, Turbopuffer, MongoDB Atlas, Vertex AI, Azure AI Search) need an account, an API key, and one region decision. Theory (HNSW vs IVF vs DiskANN vs ScaNN) is covered in Report 03 [1]; for this report it only matters that **you usually do not pick the index by hand** — the backend's default is fine.
2. **"Do you need a vector DB at all?"** If your corpus is ≤100k vectors, lives on one machine, fits in RAM, and you query from one process, a NumPy array with `np.dot` or `sklearn.neighbors.NearestNeighbors` or a FAISS flat index is the right answer. You start needing a real vector DB when **any** of these become true: data must persist across restarts, multiple processes/containers must query it, you need metadata filtering at scale (>100k rows), you need hybrid (keyword + vector) ranking, you need multi-tenant isolation (namespaces), you need >1k QPS, or you need bulk upserts and queries interleaved without rebuilding an index.
3. **Free tiers worth knowing (May 2026, all volatile — verify via the linked pricing page before quoting in code):** Pinecone Starter — 2 GB storage, up to 5 serverless indexes, up to 100 namespaces/index, 2M write units & 1M read units/month, AWS us-east-1 only, indexes pause after 3 weeks of inactivity [4][5]. Qdrant Cloud — 0.5 vCPU / 1 GB RAM / 4 GB disk, permanent, no card [6]. Weaviate Cloud — 14-day sandbox only, no permanent free managed tier; OSS self-host under BSD-3 is free [10][11]. MongoDB Atlas M0 — 512 MB, permanent [40]. Atlas Flex (GA Feb 2025) starts at $8/month with a hard cap of exactly $30/month per the official GA blog [41]. Turbopuffer — usage-based, no permanent free tier but the per-query/storage rates are explicitly "10× cheaper" than peers [42]. Azure AI Search — a "Free" tier (~50 MB, one per subscription, no semantic ranker) [54]. Vertex AI Vector Search — no free tier, but new GCP accounts get $300 credit [53].
4. **Vendor lock-in cost — ranked easiest-to-hardest to leave (with the realistic "what is mechanical vs what is not"):**
   - **Easy:** Chroma, LanceDB (Parquet/Arrow on disk — copy the folder), pgvector (`pg_dump`), Qdrant (snapshot API, `scroll` for streaming export), DuckDB-VSS, sqlite-vec (the database is a file).
   - **Medium:** Milvus (`milvus-backup` tool), Weaviate (`/backups` REST), Elasticsearch/OpenSearch (`_reindex` to remote, snapshot/restore), Redis (RDB dump but vector schema must be recreated).
   - **Hard:** Pinecone — no native "dump everything" API; you must `list` IDs then `fetch` in batches (the 2024-10 import/export API helps for cross-region within Pinecone but is not a portable dump format); Turbopuffer and Vertex AI Vector Search are similar (scroll/list and re-upsert). The portable interchange formats that actually work are **JSONL of `{id, vector, text, metadata}`** for simplicity and **Parquet with `FixedSizeList<Float32, dim>`** for size and speed.
   - **Effectively irreversible decisions** (so the facade should make the user pick them up front, not hide them): **vector dimension**, **distance metric** (cosine vs L2 vs dot product — Pinecone, Qdrant, and pgvector all let you pick once per index/collection), and **ID type** (Pinecone wants strings, Qdrant accepts UUIDs or unsigned ints, Weaviate wants UUIDs, some backends require autoincrementing ints).
5. **Hybrid search and "the DB embeds for me" are the two features a facade most often has to abstract poorly.** Native hybrid (BM25 + vector with one query): Weaviate, Elasticsearch (RRF retriever, 8.8+), Vespa, Redis 8 (RediSearch HybridQuery, 8.4.0+), Pinecone (sparse + dense, 2025), Qdrant (sparse vectors), Vertex AI Vector Search 2.0, Azure AI Search. Not native: Chroma (separate full-text path), pgvector (you write the SQL by hand, joining `ts_rank` and `<=>` ), FAISS, sqlite-vec, LanceDB (BM25 since 2024 but a separate query).
6. **License changes in the last 12 months you must flag in any backend registry:** Redis 8 → AGPLv3/SSPLv1/RSALv2 [55], with the Valkey BSD fork as an alternative; Elastic → AGPLv3 added [56]; pgvecto.rs deprecated → VectorChord (dual AGPLv3 / TensorChord Elastic License) [49][50]; Marqo OSS deprecated [52]. Apache-2.0 unchanged: Qdrant, Milvus, Chroma, LanceDB, Lance, FAISS (MIT + BSD-3), pgvector (PostgreSQL license / "permissive PG-style"), DuckDB (MIT). BSD-3: Weaviate.

## Details

### §1. Scope and how to use this report

A vector database stores high-dimensional vectors (embeddings produced by models such as those in Report 02) and answers "give me the *k* nearest vectors to this one", typically with metadata filtering and increasingly with hybrid lexical+vector ranking. **Three deployment archetypes dominate the setup question and you should pick one before anything else:**

- **Embedded / in-process** — `pip install`, no server. Database lives in your Python process or in a single file on disk. Backends: Chroma local (PersistentClient) [19][20], LanceDB [21][22], sqlite-vec [24][25], DuckDB-VSS [28], FAISS (technically a library, not a DB) [45], Milvus Lite [18].
- **Self-hosted server** — a process or container you operate. Backends: Qdrant [7][9], Weaviate [12][14], Milvus Standalone [16][17], Redis 8 + RediSearch [33], Elasticsearch / OpenSearch [37][38], Vespa [47].
- **Managed cloud** — somebody else operates it. Backends: Pinecone [4], Zilliz Cloud (managed Milvus), Qdrant Cloud [6], Weaviate Cloud [10], MongoDB Atlas Vector Search [40][41], Turbopuffer [42], Vertex AI Vector Search [53], Azure AI Search [54].

Qdrant, Weaviate, Milvus, and Chroma span all three modes — same wire protocol, so prototypes migrate cleanly. **The "do you even need a vector DB?" answer:** if all of `len(corpus) < ~100_000`, `vectors.nbytes < RAM/2`, one process, no persistence needed, no metadata filtering, no concurrent writes → use NumPy + brute-force cosine, `sklearn.neighbors.NearestNeighbors`, or a FAISS flat index. Cross any of those thresholds and a vector DB starts paying for itself.

### §2. The 2026 landscape at a glance

| Provider | Deployment | License | Free tier (managed) | Embedded | Managed | Native hybrid | Filter dialect | Maturity | Sweet spot |
|---|---|---|---|---|---|---|---|---|---|
| Chroma | Embedded / Docker / Cloud | Apache-2.0 | $5 credit, Cloud GA | ✅ | ✅ (Cloud) | partial (BM25 added) | Mongo-ish `$op` | Production-leaning (1.5.x) | "fastest path to a working RAG demo" |
| LanceDB | Embedded / S3 / Cloud (Enterprise) | Apache-2.0 | OSS free | ✅ | ✅ | BM25 + vector | SQL `WHERE` | Production for OSS | "embedded + multimodal + cheap S3 store" |
| sqlite-vec | Embedded | MIT / Apache-2.0 | n/a | ✅ | ❌ | no (separate FTS5) | SQL | v0.1.9 (Mar 2026) | "I already use SQLite" [27] |
| DuckDB-VSS | Embedded | MIT | n/a | ✅ | ❌ | no (separate FTS) | SQL | Experimental (HNSW persistence is opt-in) | "analytics + ANN in the same query" [28] |
| FAISS | Library | MIT + BSD-3 | n/a | ✅ | ❌ | no | none (you filter outside) | Mature | "in-memory benchmark / baseline" [45][46] |
| Qdrant | Server / Cloud / Hybrid Cloud | Apache-2.0 | 0.5 vCPU/1 GB/4 GB, permanent | local mode in client | ✅ | ✅ (sparse + dense) | rich JSON payload filters | Production | "Rust-fast self-host with a clean Python SDK" [6][8] |
| Weaviate | Server / Cloud / BYOC | BSD-3 | 14-day sandbox only | embedded (Python) | ✅ | ✅ | GraphQL-ish + `Filter` builder | Production | "hybrid search + vectorizer modules" [12][13] |
| Milvus / Zilliz | Server / Lite / Cloud | Apache-2.0 | Zilliz Free cluster | ✅ Milvus Lite (Linux/Mac) | ✅ | ✅ | boolean expr language | Production (2.6.x) | "billion-vector + DiskANN" [16][18] |
| Redis 8 + RediSearch | Server / Cloud | AGPLv3 / SSPLv1 / RSALv2 (May 2025) [55] | Redis Cloud free tier | ❌ | ✅ | ✅ (Hybrid 8.4+) | RediSearch DSL | Production | "I already run Redis" [33][35] |
| Elasticsearch | Server / Cloud / Serverless | AGPLv3 / SSPLv1 / Elastic 2.0 [39][56] | Cloud trial | ❌ | ✅ | ✅ (RRF) | full ES query DSL | Production | "I already run Elastic, want kNN + BM25" [37][38] |
| OpenSearch | Server / AWS managed | Apache-2.0 | AWS free tier | ❌ | ✅ | ✅ | ES-compatible | Production | "AWS-native ES fork, vector capable" |
| Pinecone | Managed (BYOC preview) | Proprietary | Starter: 2 GB, 5 indexes, AWS us-east-1 [5] | ❌ | ✅ | ✅ (sparse+dense) | Mongo-ish `$op` | Production | "zero-ops managed canonical" [4] |
| MongoDB Atlas | Managed (also self-managed CE preview) | SSPLv1 (server) | M0 512 MB permanent + Flex $8–$30 [41] | ❌ | ✅ | ✅ ($search + $vectorSearch) | MongoDB Query | Production | "I already have MongoDB" [40] |
| Turbopuffer | Managed (object-storage-first) | Proprietary | Usage-based, no permanent free | ❌ | ✅ | ✅ (BM25 + vector) | typed attribute filters | Production (GA) | "10–100× cheaper, S3-backed" [42][43] |
| Vespa | Server (Docker/k8s) / Cloud | Apache-2.0 | $300 trial credit | ❌ | ✅ | ✅ (best-in-class) | YQL + rank profiles | Mature | "billion-scale hybrid with custom ranking" [47][48] |
| pgvector | Postgres extension | PG-style permissive | n/a | ✅ (any PG) | via managed PG | manual SQL hybrid | SQL `WHERE` | Production (v0.8.2, **Feb 2026**, fixes CVE-2026-3172 buffer overflow in parallel HNSW builds) [30] | "I already run Postgres" |
| VectorChord (`vchord`) | Postgres extension | AGPLv3 / TensorChord Elastic v1 | n/a | ✅ (any PG) | via managed PG | manual SQL hybrid | SQL `WHERE` | Active (successor to pgvecto.rs) | "pgvector at 10–100M+ scale, cheaper RAM" [49] |
| Vald | Kubernetes only | Apache-2.0 | n/a | ❌ | self-managed | ❌ | gRPC | Active, low velocity (v1.7.17 Jul 2025) | "k8s-native NGT at scale" [51] |
| Marqo | Managed (OSS deprecated) | Apache-2.0 (OSS, frozen) | Cloud free tier exists | ❌ | ✅ | ✅ | own filter string | OSS deprecated [52] | "ecommerce product search SaaS" |
| Vertex AI Vector Search | Managed (GCP) | Proprietary | $300 GCP credit | ❌ | ✅ | ✅ (2.0) | restrict+crowding tokens | Production | "GCP-native ScaNN" [53] |
| Azure AI Search | Managed (Azure) | Proprietary | Free tier (~50 MB) | ❌ | ✅ | ✅ + semantic ranker | OData filter | Production | "Azure-native, integrated vectorization" [54] |

### §3. Per-provider profiles (essentials only — defer theory to Report 03 [1])

#### Embedded

**Chroma** [19][20] — Python-first embedded vector DB; `pip install chromadb`. `chromadb.Client()` (in-memory) or `chromadb.PersistentClient(path="./db")` (persistent on disk). Server mode: `chroma run --path /db_path --port 8000` or `docker run -p 8000:8000 -v ./chroma-data:/data chromadb/chroma`. Same codebase powers Chroma Cloud (Apache-2.0). *Choose when:* RAG prototype, local-first AI app. *Avoid when:* you need billion-scale or sub-10ms p99 at high QPS. *Gotchas:* (a) the default embedding function is Sentence Transformers `all-MiniLM-L6-v2` — fine for demos but a 384-dim trap if you later switch to OpenAI 1536-dim; (b) telemetry is on by default — set `ANONYMIZED_TELEMETRY=False`; (c) collection names must match the regex (no spaces). *Bulk export:* iterate `collection.get(include=['embeddings','metadatas','documents'])` in pages → JSONL.

**LanceDB** [21][22][23] — embedded multimodal vector DB on the Lance columnar format; `pip install lancedb`. `db = lancedb.connect("./mydb")` for local or `lancedb.connect("s3://bucket/lancedb")` for object-storage mode. Schema evolution and zero-copy versioning are real differentiators — the storage is Arrow Parquet-like, so a `.lance` folder is portable and inspectable with `pyarrow`. *Choose when:* embedded app, multimodal (text/image/video), want cheap S3 vector store, or want versioned data. *Avoid when:* you need >100 concurrent writers from different processes (single-writer semantics).

**sqlite-vec** [24][25][26][27] — `pip install sqlite-vec`; the successor to sqlite-vss by Alex Garcia. Requires SQLite ≥3.41 for full features; on macOS the system Python's sqlite3 has extensions disabled, so use Homebrew Python or `pysqlite3-binary`. Loads as an extension: `db.enable_load_extension(True); sqlite_vec.load(db)`. Brute-force on packed BLOB vectors — no ANN index yet, but extremely fast for ≤1M vectors. *Choose when:* your app already uses SQLite. *Avoid when:* you need ANN above ~1M vectors.

**DuckDB-VSS** [28][29] — `INSTALL vss; LOAD vss;` then `CREATE INDEX … USING HNSW (vec) WITH (metric = 'cosine')`. HNSW persistence is experimental (`SET hnsw_enable_experimental_persistence = true`) — you may lose the index across restarts unless you opt in. *Choose when:* you do analytics on the same column you do ANN over. *Avoid when:* you need rock-solid persistence guarantees today.

**FAISS** [45][46] — a library, not a database. `pip install faiss-cpu` (current 1.13.2, Dec 2025, Python 3.10–3.14) or `conda install -c pytorch faiss-gpu` for GPU. The reference implementation of every ANN algorithm — `IndexFlatIP`, `IndexHNSWFlat`, `IndexIVFPQ`, etc. *Choose when:* in-memory benchmark, baseline, or component inside another system. *Avoid when:* you need filtering, persistence, multi-process, or a server — those are out of scope for FAISS.

#### Self-hosted server (also-managed)

**Qdrant** [6][7][8][9] — Rust-written, payload-rich, supports scroll, quantization (scalar/binary/product), sparse vectors, and a `local mode` in the Python client. `pip install qdrant-client`. One-liner: `docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant`. Pin client major-version to server major-version (e.g. `qdrant-client~=1.13`). Standout: rich JSON payload filters, named vectors, scroll API, quantization. *Choose when:* you want a single backend that works embedded (in-memory or local disk via the client), as Docker, on Kubernetes, or managed, with the same API. *Pricing:* free 0.5 vCPU/1 GB/4 GB forever; Standard is hourly usage-based (~$0.078/GB-hour); Hybrid Cloud puts the data plane in your VPC [6].

**Weaviate** [10][11][12][13][14][15] — BSD-3, GraphQL-and-gRPC API, modular vectorizer ("you give me text, I call OpenAI/Cohere/HuggingFace/Ollama and embed for you"), excellent hybrid (BM25+vector) baked in. `pip install -U weaviate-client` (v4.x). v4 GA is **strictly more typed** than v3 — `Configure.Vectors.text2vec_openai()`, `Property(name=…, data_type=DataType.TEXT)`. **2025–2026 changes you must know:** (1) the v3 client is deprecated; (2) Configure.NamedVectors → Configure.Vectors (v4.16.0+); (3) Cloud pricing restructured October 27 2025 (per the official Weaviate blog post) into Sandbox (14 days only, cannot extend) / Flex ($45/mo minimum, pay-as-you-go) / Plus ($280/mo, annual commitment) / Premium ($400/mo) / Enterprise BYOC [10][11]. Don't believe any older "$25/month" figure.

**Milvus / Milvus Lite / Zilliz Cloud** [16][17][18] — Apache-2.0, written in Go/C++, best at billion-vector scale (DiskANN on SSD instead of RAM). Three install modes:
- *Milvus Lite* (Linux/macOS only — **not Windows-native**, use WSL2): `pip install -U "pymilvus[milvus_lite]"` → `MilvusClient("./milvus_demo.db")`. Up to ~1M vectors, no auth, single-process file lock [18].
- *Standalone* via the official script: `curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh && bash standalone_embed.sh start` (port 19530, WebUI on 9091) [16].
- *Distributed* (k8s, Helm) or *Zilliz Cloud*.

**Redis 8 + RediSearch (Redis Query Engine)** [33][34][35][36] — since Redis 8 (May 2025), RediSearch/JSON/Bloom are in core. Licensing: AGPLv3 / SSPLv1 / RSALv2 — you pick [55]. `docker run -p 6379:6379 redis/redis-stack:latest` or `redis:8` (8 includes the modules natively). Schema: `FT.CREATE idx ON HASH PREFIX 1 doc: SCHEMA embedding VECTOR HNSW 10 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE`. New HybridQuery (8.4.0+) combines BM25 and vector with `LINEAR` or `RRF`. Python: `pip install redis redisvl`. *Choose when:* you already run Redis or need sub-ms KNN on a hot dataset. *Gotchas:* HNSW index is in RAM — size accordingly; updates are heavier than point writes.

**Elasticsearch / OpenSearch** [37][38][39] — kNN via `dense_vector` field with `int8_hnsw` as the default index type since 8.x, `bbq_hnsw` (binary quantization) as the default for ≥384-dim vectors in 8.18+. RRF retriever for hybrid (BM25 + kNN) is GA in 8.8+. `pip install elasticsearch` or `elasticsearch-dsl`. License: Elastic added AGPLv3 alongside SSPLv1 and Elastic License 2.0 in August 2024 [56]. *Choose when:* you already operate Elastic/OpenSearch and want hybrid; *avoid when:* you don't already run it — the operational overhead is real.

#### Managed-first

**Pinecone** [1][2][3][4][5] — fully managed, serverless-first since 2024. **Critical 2024–2026 changes:** (1) package renamed `pinecone-client` → `pinecone` at v5.1.0; the v6/v7/v8/v9 series follow, with **v9.0.0 (May 2026)** requiring Python ≥3.10 [1][2]; (2) pod-based indexes are legacy — new indexes are serverless and billed per read unit / write unit / storage; (3) integrated inference (Pinecone embeds for you with `create_index_for_model`); (4) Bring Your Own Cloud (BYOC) for AWS in public preview. *Starter (free) tier limits as of May 2026* per the official docs [5]: 2 GB storage, up to 5 serverless indexes, up to 100 namespaces/index, 2M WUs and 1M RUs per month, AWS us-east-1 only, indexes pause after 3 weeks of inactivity, 1 project, 2 users/org. Standard plan has a $50/month minimum commitment.

**MongoDB Atlas Vector Search** [40][41] — `$vectorSearch` aggregation stage. M0 free (512 MB, permanent). The **Atlas Flex** tier went GA February 6 2025 and combines and replaces the legacy Shared (M2/M5) and Serverless tiers — base $8/month, **capped at exactly $30/month per the official GA blog**, supports Vector Search and Atlas Search [41]. As of 2025 MongoDB also added Vector Search to the self-managed Community Edition in public preview, lowering the air-gapped barrier. Auto Embeddings (Voyage AI) can embed for you. Limits: M0 supports only one vector index.

**Turbopuffer** [42][43][44] — managed, object-storage-first (vectors live on S3/GCS, hot tier in memory/SSD cache), 10–100× cheaper than the per-GB-hour competitors. Region-pinned (e.g. `gcp-us-central1`). `pip install turbopuffer` (current stable **v2.1.0**, released May 17 2026 per PyPI). API is "namespaces of rows", each with `id`, `vector`, attributes. Hybrid search (BM25 + vector) GA. *Choose when:* huge-but-bursty workload, cost matters more than sub-50ms latency. *Avoid when:* you need air-gapped or sub-10ms p50.

#### Short profiles ("here is when you'll actually need this")

- **Vespa** [47][48] — Apache-2.0, Yahoo-grade hybrid + ML-rank engine. `pip install pyvespa`; `docker run --name vespa -p 8080:8080 -p 19071:19071 vespaengine/vespa`. Vespa Cloud free trial advertises $300 credit. Reach for it at billion-scale with custom learning-to-rank.
- **pgvecto.rs / VectorChord (`vchord`)** [49][50] — TensorChord's PostgreSQL extensions. **pgvecto.rs is deprecated**; the README itself directs you to VectorChord (RaBitQ + hierarchical k-means, dual-licensed AGPLv3 / TensorChord Elastic v1). Quickstart: `docker run -e POSTGRES_PASSWORD=... -p 5432:5432 tensorchord/vchord-postgres:pg17-v0.1.0` → `CREATE EXTENSION vchord CASCADE;`.
- **Vald** [51] — Apache-2.0, Yahoo Japan, Kubernetes-only (Helm), NGT-based. Latest v1.7.17 (Jul 2025). Pick only if you have a k8s platform team.
- **Marqo** [52] — verbatim GitHub README notice: *"Marqo's Open Source project is deprecated and will no longer receive updates."* Code remains Apache-2.0 but no further OSS releases. Use the managed Marqo Cloud only if ecommerce product search is your specific use case.
- **Vertex AI Vector Search** [53] — Google Cloud's managed ScaNN. `pip install google-cloud-aiplatform`. Per-node-hour billing for the deployed index endpoint (idle still costs). 2.0 added auto-embeddings and hybrid.
- **Azure AI Search** [54] — `pip install azure-search-documents`. Free tier (~50 MB, no semantic ranker) is permanent and one-per-subscription; vector capability is included at all tiers including Free.

#### pgvector (full profile because "I already run Postgres" is the single most common situation) [30][31][32]

*What it is:* a PostgreSQL extension that adds a `vector(n)` data type, three distance operators (`<->` L2, `<#>` negative inner product, `<=>` cosine), HNSW and IVFFlat index access methods, half-precision (`halfvec`), binary (`bit`), and sparse (`sparsevec`) variants. Stable v0.8.2 was released **February 26 2026** and notably fixes **CVE-2026-3172**, a buffer overflow in parallel HNSW index builds — *upgrade immediately if you use parallel index builds*. (The CVE was disclosed in the official PostgreSQL News announcement.)

*Deployment:* any Postgres ≥13. Self-host: build from source (`make && make install`) or `apt install postgresql-NN-pgvector`; managed: AWS RDS, GCP Cloud SQL, Azure, Neon, Supabase, Timescale, Crunchy all support it. Docker: `pgvector/pgvector:pg17`.

*Hello world:*
```sql
CREATE EXTENSION vector;
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(1536));
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);
SELECT * FROM items ORDER BY embedding <=> '[0.1,0.2,...]'::vector LIMIT 5;
```

*Python:* `pip install pgvector psycopg[binary]` then `from pgvector.psycopg import register_vector; register_vector(conn)`. Also bindings for psycopg2, asyncpg, SQLAlchemy, Django, peewee. *Pricing & free tier:* free (it's a Postgres extension); cost is your Postgres host. *Lock-in:* trivial (`pg_dump`). *Pros:* relational joins with vectors in one SQL query; mature transactions, backups, replication, monitoring. *Cons:* HNSW index lives off-heap and is rebuilt on `pg_restore`; high-write workloads can be slower than purpose-built DBs at >50M vectors — that's when to consider VectorChord [49]. *Gotchas:* (1) `vector(N)` enforces the dimension on insert; mismatched models silently fail with type errors; (2) HNSW index *build* uses `maintenance_work_mem` — set it generously; (3) the AWS RDS pgvector version often lags upstream by ~6 months.

### §4. The decision framework (directly translatable to a recommender function)

```
Q1. Corpus size?
  ≤100k → "Do you even need a DB?" path → NumPy / sklearn / FAISS flat. Stop.
  100k–10M → embedded or single-server is fine.
  10M–100M → managed or beefy self-host with quantization.
  >100M → Milvus, Vespa, Pinecone, Turbopuffer.

Q2. Do you need persistence?
  No (CI test, notebook) → in-memory FAISS or Chroma(Client()) or Qdrant(":memory:").
  Yes → continue.

Q3. Can the user run Docker / a server?
  No → embedded only (Chroma persistent, LanceDB, sqlite-vec, DuckDB-VSS, Milvus Lite, FAISS).
  Yes → Q4.

Q4. Cloud allowed, or air-gapped / on-prem?
  Air-gapped → Qdrant self-host, Milvus standalone, Weaviate self-host, pgvector, VectorChord, Redis self-host, Elasticsearch self-host, Chroma, LanceDB (local mode). Disable Chroma telemetry.
  Cloud allowed → Q5.

Q5. Budget — free only, or paid OK?
  Free only → Qdrant Cloud free (permanent 1 GB) > Pinecone Starter > MongoDB M0 > Azure AI Search Free > Zilliz free cluster. Avoid Weaviate Cloud here — only 14-day sandbox.
  Paid OK → Q6.

Q6. Already running Postgres / Redis / Elastic / Mongo?
  Postgres → pgvector (or VectorChord at >10M). Don't add another DB.
  Redis → Redis 8 native vector. Mind the AGPL/RSAL license change.
  Elastic/OpenSearch → use kNN + RRF in-place.
  Mongo → Atlas Vector Search ($vectorSearch).
  None → Q7.

Q7. Need hybrid (keyword + vector) in one query?
  Yes → Weaviate, Elasticsearch (RRF), Vespa, Redis 8 (HybridQuery 8.4+), Pinecone (sparse+dense), Turbopuffer.
  No → simpler choices like Qdrant, Chroma, Pinecone work fine.

Q8. Multi-tenant / namespaces?
  Yes → Pinecone (namespaces), Weaviate (multi-tenancy collections), Qdrant (collection-per-tenant or payload filter), Turbopuffer (namespace = first-class).

Q9. QPS and write pattern?
  Bulk-once, light query → almost anything works.
  Continuous writes + sub-50ms p99 reads → Pinecone, Qdrant (with quantization), Vespa, Turbopuffer.
  >10k QPS → Pinecone Dedicated Read Nodes, Vespa, Vald, Zilliz dedicated.
```

**Rubric: situation → first pick / runner-up / why**

| Situation | First pick | Runner-up | Why |
|---|---|---|---|
| Throwaway prototype in a notebook | Chroma in-memory | Qdrant `":memory:"` | one pip install, no server |
| Local desktop app that must persist | LanceDB | Chroma PersistentClient | single folder on disk, no daemon |
| ~500k docs, one server, no ops team | Qdrant Docker | Chroma Docker | best perf/RAM with a clean Python SDK |
| Millions of vectors, need managed | Pinecone Serverless | Qdrant Cloud / Turbopuffer | zero ops, predictable scaling |
| Already on Postgres | pgvector | VectorChord | no second DB; SQL joins for free |
| Air-gapped / on-prem | Qdrant or Milvus standalone | pgvector | Apache-2.0, no telemetry, well-documented k8s path |
| Needs hybrid (keyword+vector) search | Weaviate | Elasticsearch + RRF | native hybrid is best-in-class |
| Needs to scale to billions | Milvus / Zilliz | Vespa | DiskANN keeps RAM bounded |
| Cost-sensitive, bursty, OK with S3 latency | Turbopuffer | LanceDB on S3 | object-storage economics |

**Anti-recommendations.** Do not pick Pinecone if you need a portable bulk export and may want to leave — exit is real work. Do not pick Weaviate Cloud for an indefinite free dev environment — the sandbox is 14 days only and cannot be extended; self-host instead. Do not pick MongoDB Atlas M0 for vector search if you need >1 vector index (M0 limit). Do not pick Milvus Lite on Windows (Linux/macOS only — use WSL2). Do not pick Vald if you don't already operate Kubernetes. Do not start a new project on `pgvecto.rs` — go straight to VectorChord. Do not start a new project on Marqo OSS — it is officially deprecated.

**Revisit benchmarks** before committing on your own corpus: `recall@10` against a brute-force ground truth, p95 query latency under target QPS, ingest throughput (vectors/sec), `$ / month` at projected scale (1M, 10M, 100M vectors with your dimension).

### §5. Installation & setup playbooks (the practitioner payload)

Throughout: **never commit API keys.** Use environment variables. Conventional names: `PINECONE_API_KEY`, `WEAVIATE_API_KEY` + `WEAVIATE_URL`, `QDRANT_API_KEY` + `QDRANT_URL`, `TURBOPUFFER_API_KEY`, `OPENAI_API_KEY`, `MONGODB_URI`.

**Chroma** [19][20]
```bash
pip install chromadb            # current 1.5.x as of May 2026
# Server mode (also works):
docker run -p 8000:8000 -v ./chroma-data:/data chromadb/chroma
```
Hello-world (embedded):
```python
import chromadb
client = chromadb.PersistentClient(path="./db")
col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
col.add(ids=["a","b","c"],
        documents=["the cat sat","the dog ran","fresh pizza"],
        metadatas=[{"k":"animal"},{"k":"animal"},{"k":"food"}])
print(col.query(query_texts=["pet"], n_results=2, where={"k":"animal"}))
```
*Pitfalls:* default embedder is 384-dim; macOS Python sometimes hits SQLite extension limits → use Homebrew Python; telemetry → `os.environ["ANONYMIZED_TELEMETRY"]="False"` *before* the import.

**LanceDB** [21][22]
```bash
pip install lancedb
```
```python
import lancedb, numpy as np
db = lancedb.connect("./mydb")
tbl = db.create_table("docs", data=[
    {"id":"a", "vector": np.random.rand(384).astype("float32").tolist(), "text":"the cat sat"},
    {"id":"b", "vector": np.random.rand(384).astype("float32").tolist(), "text":"the dog ran"},
])
print(tbl.search(np.random.rand(384).astype("float32").tolist()).limit(2).to_list())
```
*Pitfalls:* a `.lance` directory is a single-writer dataset — coordinate writes; on S3 set `LANCE_*` AWS env vars.

**sqlite-vec** [24][25]
```bash
pip install sqlite-vec
```
```python
import sqlite3, sqlite_vec, struct
db = sqlite3.connect(":memory:")
db.enable_load_extension(True); sqlite_vec.load(db); db.enable_load_extension(False)
db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")
db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", [struct.pack("4f", 0.1,0.2,0.3,0.4)])
print(db.execute("SELECT rowid, distance FROM v WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
                 [struct.pack("4f", 0.1,0.2,0.3,0.4)]).fetchall())
```
*Pitfalls:* macOS system Python disables extensions — use `brew install python` or pysqlite3-binary; SQLite ≥3.41 strongly recommended.

**DuckDB-VSS** [28]
```bash
pip install duckdb
```
```python
import duckdb
con = duckdb.connect("vec.duckdb")
con.execute("INSTALL vss; LOAD vss;")
con.execute("CREATE TABLE items(id INTEGER, embedding FLOAT[3])")
con.execute("INSERT INTO items VALUES (1,[0.1,0.2,0.3]),(2,[0.4,0.5,0.6])")
con.execute("CREATE INDEX idx ON items USING HNSW (embedding) WITH (metric='cosine')")
print(con.execute("SELECT id FROM items ORDER BY array_cosine_distance(embedding, [0.1,0.2,0.3]::FLOAT[3]) LIMIT 1").fetchall())
```
*Pitfalls:* persistent HNSW is opt-in — `SET hnsw_enable_experimental_persistence=true;`.

**FAISS** [45][46]
```bash
pip install faiss-cpu          # current 1.13.2 (Dec 2025), Python 3.10-3.14
# Or GPU:
# conda install -c pytorch -c nvidia faiss-gpu=1.14.1
```
```python
import faiss, numpy as np
xb = np.random.random((10000,128)).astype("float32")
index = faiss.IndexFlatL2(128); index.add(xb)
D,I = index.search(np.random.random((1,128)).astype("float32"), 5)
```

**Qdrant** [7][8][9]
```bash
pip install qdrant-client     # pin to server major: ~=1.13
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
curl http://localhost:6333/healthz       # verify
```
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
c = QdrantClient(":memory:")            # or QdrantClient(url="http://localhost:6333")
c.create_collection("test", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
c.upsert("test", points=[
    PointStruct(id=1, vector=[0.05,0.61,0.76,0.74], payload={"city":"Berlin"}),
    PointStruct(id=2, vector=[0.19,0.81,0.75,0.11], payload={"city":"London"}),
])
print(c.query_points("test", query=[0.2,0.1,0.9,0.7], limit=1).points)
```
*Cloud signup:* cloud.qdrant.io → create cluster (free tier) → copy URL + API key → `QdrantClient(url=URL, api_key=KEY)`. *Pitfalls:* Windows volume mounting needs a named Docker volume; default has no auth — set `QDRANT__SERVICE__API_KEY` for any exposed instance.

**Weaviate** [12][13][14]
```bash
pip install -U weaviate-client          # v4.x (v4.21+ as of May 2026)
docker run -p 8080:8080 -p 50051:50051 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true cr.weaviate.io/semitechnologies/weaviate:latest
```
```python
import weaviate
from weaviate.classes.config import Property, DataType, Configure
with weaviate.connect_to_local() as client:
    client.collections.create("Article",
        properties=[Property(name="title", data_type=DataType.TEXT)],
        vector_config=Configure.Vectors.self_provided())
    col = client.collections.use("Article")
    col.data.insert({"title":"Hello"}, vector=[0.1]*768)
    print(col.query.near_vector([0.1]*768, limit=2).objects)
```
*Cloud signup:* console.weaviate.cloud → create sandbox (14 days only) → `weaviate.connect_to_weaviate_cloud(cluster_url=..., auth_credentials=Auth.api_key(...))`. *Pitfalls:* (1) you must `client.close()` or use the context manager since v4.4b7; (2) Configure.NamedVectors renamed to Configure.Vectors at v4.16.0; (3) v3 client is deprecated — migrate.

**Milvus / Milvus Lite** [16][17][18]
```bash
# Milvus Lite (embedded; Linux/macOS only):
pip install -U "pymilvus[milvus_lite]"
# Standalone (Docker, all platforms):
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start          # port 19530, WebUI on 9091
```
```python
from pymilvus import MilvusClient
client = MilvusClient("./milvus_demo.db")          # Lite — or MilvusClient(uri="http://localhost:19530")
client.create_collection(collection_name="demo", dimension=8)
client.insert("demo", [{"id":1, "vector":[0.1]*8, "city":"Berlin"}])
print(client.search("demo", data=[[0.1]*8], limit=1, output_fields=["city"]))
```
*Pitfalls:* (1) Lite is **not available on native Windows** — use WSL2; (2) on macOS allocate Docker ≥2 vCPU + 8 GB RAM; (3) older `milvus-lite` (C++/CGo) and the new pure-Python rewrite have incompatible `.db` formats — back up before upgrading.

**Redis 8 + RediSearch** [33][34][35]
```bash
docker run -p 6379:6379 redis:8     # or redis/redis-stack:latest on Redis 7.x
pip install redis redisvl
```
```python
import numpy as np, redis
from redis.commands.search.field import VectorField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
r = redis.Redis()
r.ft("idx").create_index(
    [TagField("tag"),
     VectorField("v","HNSW",{"TYPE":"FLOAT32","DIM":4,"DISTANCE_METRIC":"COSINE"})],
    definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH))
r.hset("doc:1", mapping={"tag":"a","v":np.array([0.1,0.2,0.3,0.4],"float32").tobytes()})
```
*Pitfalls:* license change (AGPLv3/SSPLv1/RSALv2 since Redis 8) — verify with legal; HNSW lives in RAM.

**Elasticsearch** [37][38]
```bash
pip install elasticsearch
docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.18.0
```
```python
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")
es.indices.create(index="docs", mappings={"properties":{"v":{"type":"dense_vector","dims":4,"similarity":"cosine"}}})
es.index(index="docs", document={"v":[0.1,0.2,0.3,0.4], "t":"hello"})
es.indices.refresh(index="docs")
print(es.search(index="docs", knn={"field":"v","query_vector":[0.1,0.2,0.3,0.4],"k":1,"num_candidates":10}))
```

**pgvector** [30][31][32]
```bash
# In Postgres (Docker shortcut):
docker run -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg17
pip install "psycopg[binary]" pgvector
```
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE items (id BIGSERIAL PRIMARY KEY, embedding vector(1536), tag TEXT);
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);
```
```python
import psycopg
from pgvector.psycopg import register_vector
conn = psycopg.connect("postgresql://postgres:pw@localhost:5432/postgres")
register_vector(conn)
conn.execute("INSERT INTO items (embedding, tag) VALUES (%s, %s)", ([0.1]*1536, "demo"))
print(conn.execute("SELECT id FROM items ORDER BY embedding <=> %s LIMIT 5", ([0.1]*1536,)).fetchall())
```
*Upgrade urgency:* pgvector 0.8.2 (Feb 2026) fixes **CVE-2026-3172**, a buffer overflow in parallel HNSW index builds; upgrade.

**Pinecone** [1][2][3][4]
```bash
pip uninstall pinecone-client          # critical — old package name
pip install pinecone                    # current v9.x as of May 2026, Python ≥3.10
```
```python
import os
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc.create_index(name="demo", dimension=1536, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
idx = pc.Index("demo")
idx.upsert([("a",[0.1]*1536,{"k":"v"})])
print(idx.query(vector=[0.1]*1536, top_k=1, include_metadata=True))
```
*Cloud signup:* app.pinecone.io → create project → API key → set `PINECONE_API_KEY`. *Pitfalls:* don't have both `pinecone-client` and `pinecone` installed.

**MongoDB Atlas Vector Search** [40][41]
```bash
pip install pymongo
```
Create an M0 (free) or Flex cluster in the Atlas UI; create a *vector search index* on a collection with field `embedding` (knnVector, 1536, cosine). Then:
```python
import os
from pymongo import MongoClient
col = MongoClient(os.environ["MONGODB_URI"])["db"]["docs"]
col.insert_one({"text":"hello","embedding":[0.1]*1536})
print(list(col.aggregate([{"$vectorSearch":{"index":"vec_idx","path":"embedding",
    "queryVector":[0.1]*1536,"numCandidates":50,"limit":3}}])))
```

**Turbopuffer** [42][43][44]
```bash
pip install turbopuffer                 # current stable v2.1.0 (May 17, 2026)
```
```python
import os
from turbopuffer import Turbopuffer
tpuf = Turbopuffer(region="gcp-us-central1", api_key=os.environ["TURBOPUFFER_API_KEY"])
ns = tpuf.namespaces.write(namespace="demo", distance_metric="cosine_distance",
    upsert_rows=[{"id":"a","vector":[0.1,0.2],"attributes":{"name":"Red boots"}}])
```

### §6. Machine-usable provider metadata (lift into a backend registry)

```yaml
- name: chroma
  display_name: Chroma
  deployment: [embedded, server, managed]
  requires_server: false
  pip_packages: [chromadb]
  extras_suggestion: null
  managed_free_tier: true
  license: Apache-2.0
  embedded_mode: true
  hybrid_search: partial
  filter_dialect: mongo-ish ($eq, $in, $and)
  bulk_export: paginated collection.get to JSONL
  docs:
    install: https://pypi.org/project/chromadb/
    python_client: https://www.trychroma.com/products/chromadb
    pricing: https://www.trychroma.com/products/chromadb
  verify_command: "python -c 'import chromadb; c=chromadb.Client(); print(c.heartbeat())'"
  notes: "Telemetry on by default; set ANONYMIZED_TELEMETRY=False."

- name: lancedb
  display_name: LanceDB
  deployment: [embedded, server, managed]
  requires_server: false
  pip_packages: [lancedb]
  extras_suggestion: "[clip, embeddings]"
  managed_free_tier: true
  license: Apache-2.0
  embedded_mode: true
  hybrid_search: true
  filter_dialect: SQL WHERE
  bulk_export: copy .lance directory or to_pandas
  docs:
    install: https://docs.lancedb.com/quickstart
    python_client: https://lancedb.github.io/lancedb/
    pricing: https://lancedb.com
  verify_command: "python -c 'import lancedb; db=lancedb.connect(\"/tmp/lv\"); print(db.table_names())'"
  notes: "Single-writer per dataset; S3/GCS native."

- name: sqlite_vec
  display_name: sqlite-vec
  deployment: [embedded]
  requires_server: false
  pip_packages: [sqlite-vec]
  extras_suggestion: null
  managed_free_tier: false
  license: MIT-OR-Apache-2.0
  embedded_mode: true
  hybrid_search: false
  filter_dialect: SQL
  bulk_export: sqlite .dump
  docs:
    install: https://alexgarcia.xyz/sqlite-vec/installation.html
    python_client: https://alexgarcia.xyz/sqlite-vec/python.html
    pricing: null
  verify_command: "python -c 'import sqlite3,sqlite_vec; d=sqlite3.connect(\":memory:\"); d.enable_load_extension(True); sqlite_vec.load(d); print(d.execute(\"select vec_version()\").fetchone())'"
  notes: "SQLite ≥3.41; macOS system python disables extensions."

- name: duckdb_vss
  display_name: DuckDB-VSS
  deployment: [embedded]
  requires_server: false
  pip_packages: [duckdb]
  extras_suggestion: null
  managed_free_tier: false
  license: MIT
  embedded_mode: true
  hybrid_search: false
  filter_dialect: SQL
  bulk_export: COPY ... TO 'file.parquet'
  docs:
    install: https://duckdb.org/docs/current/core_extensions/vss
    python_client: https://duckdb.org/docs/current/api/python/overview
    pricing: null
  verify_command: "python -c 'import duckdb; c=duckdb.connect(); c.execute(\"INSTALL vss; LOAD vss;\"); print(\"ok\")'"
  notes: "HNSW persistence is experimental; opt in explicitly."

- name: faiss
  display_name: FAISS
  deployment: [embedded]
  requires_server: false
  pip_packages: [faiss-cpu]
  extras_suggestion: faiss-gpu (conda)
  managed_free_tier: false
  license: MIT-AND-BSD-3-Clause
  embedded_mode: true
  hybrid_search: false
  filter_dialect: none
  bulk_export: faiss.write_index
  docs:
    install: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
    python_client: https://github.com/facebookresearch/faiss/wiki/Python-interface
    pricing: null
  verify_command: "python -c 'import faiss; print(faiss.__version__)'"
  notes: "Library, not a DB. No filtering, no persistence semantics."

- name: qdrant
  display_name: Qdrant
  deployment: [embedded, server, managed]
  requires_server: false
  pip_packages: [qdrant-client]
  extras_suggestion: "[fastembed]"
  managed_free_tier: true
  license: Apache-2.0
  embedded_mode: true
  hybrid_search: true
  filter_dialect: payload filters (must/should/must_not)
  bulk_export: snapshot API + scroll
  docs:
    install: https://qdrant.tech/documentation/quickstart/
    python_client: https://python-client.qdrant.tech/
    filtering: https://qdrant.tech/documentation/concepts/filtering/
    pricing: https://qdrant.tech/pricing/
  verify_command: "curl -fsS http://localhost:6333/healthz"
  notes: "Permanent free 0.5 vCPU/1 GB cloud cluster."

- name: weaviate
  display_name: Weaviate
  deployment: [embedded, server, managed]
  requires_server: false
  pip_packages: [weaviate-client]
  extras_suggestion: null
  managed_free_tier: false   # 14-day sandbox only
  license: BSD-3-Clause
  embedded_mode: true
  hybrid_search: true
  filter_dialect: typed Filter builder / GraphQL
  bulk_export: backup/restore API
  docs:
    install: https://docs.weaviate.io/weaviate/quickstart
    python_client: https://docs.weaviate.io/weaviate/client-libraries/python
    pricing: https://weaviate.io/pricing
  verify_command: "python -c 'import weaviate; c=weaviate.connect_to_local(); print(c.is_ready()); c.close()'"
  notes: "v3 deprecated; v4.16+ renamed NamedVectors to Vectors; Oct 2025 pricing restructure (Flex $45/mo)."

- name: milvus
  display_name: Milvus / Zilliz
  deployment: [embedded, server, managed]
  requires_server: false       # Lite is embedded
  pip_packages: [pymilvus]
  extras_suggestion: "[milvus_lite]"
  managed_free_tier: true      # Zilliz free cluster
  license: Apache-2.0
  embedded_mode: true          # Lite, Linux/macOS only
  hybrid_search: true
  filter_dialect: boolean expression language
  bulk_export: milvus-backup tool
  docs:
    install: https://milvus.io/docs/install_standalone-docker.md
    python_client: https://milvus.io/docs/install-pymilvus.md
    pricing: https://zilliz.com/pricing
  verify_command: "curl -fsS http://localhost:9091/healthz"
  notes: "Milvus Lite NOT on native Windows; use WSL2."

- name: redis
  display_name: Redis 8 (RediSearch)
  deployment: [server, managed]
  requires_server: true
  pip_packages: [redis, redisvl]
  extras_suggestion: null
  managed_free_tier: true
  license: AGPLv3-OR-SSPLv1-OR-RSALv2
  embedded_mode: false
  hybrid_search: true
  filter_dialect: RediSearch DSL
  bulk_export: RDB dump + schema rebuild
  docs:
    install: https://redis.io/docs/latest/operate/oss_and_stack/install/
    python_client: https://github.com/redis/redis-vl-python
    filtering: https://redis.io/docs/latest/develop/ai/search-and-query/query/vector-search/
    pricing: https://redis.io/pricing/
  verify_command: "redis-cli ping"
  notes: "License changed May 2025; modules now in core."

- name: elasticsearch
  display_name: Elasticsearch
  deployment: [server, managed]
  requires_server: true
  pip_packages: [elasticsearch]
  extras_suggestion: elasticsearch-dsl
  managed_free_tier: true       # Elastic Cloud trial
  license: AGPLv3-OR-SSPLv1-OR-Elastic-2.0
  embedded_mode: false
  hybrid_search: true
  filter_dialect: ES query DSL
  bulk_export: snapshot/restore, _reindex remote
  docs:
    install: https://www.elastic.co/docs/deploy-manage/deploy/self-managed
    python_client: https://elasticsearch-py.readthedocs.io/
    filtering: https://www.elastic.co/docs/solutions/search/vector/knn
    pricing: https://www.elastic.co/pricing/
  verify_command: "curl -fsS http://localhost:9200/"
  notes: "AGPLv3 added Aug 2024 alongside SSPLv1 and ELv2."

- name: pinecone
  display_name: Pinecone
  deployment: [managed]
  requires_server: false
  pip_packages: [pinecone]      # NOT pinecone-client (deprecated)
  extras_suggestion: "[asyncio, grpc]"
  managed_free_tier: true
  license: Proprietary
  embedded_mode: false
  hybrid_search: true
  filter_dialect: mongo-ish ($eq, $in)
  bulk_export: list IDs + fetch in batches; 2024-10 import/export API (Pinecone↔Pinecone)
  docs:
    install: https://pypi.org/project/pinecone/
    python_client: https://sdk.pinecone.io/python/
    filtering: https://docs.pinecone.io/guides/data/filtering-with-metadata
    pricing: https://www.pinecone.io/pricing/
    limits: https://docs.pinecone.io/reference/api/database-limits
  verify_command: "python -c 'from pinecone import Pinecone; print(Pinecone().list_indexes())'"
  notes: "Package renamed at v5.1.0; v9 requires Python ≥3.10."

- name: mongodb_atlas
  display_name: MongoDB Atlas Vector Search
  deployment: [server, managed]
  requires_server: true
  pip_packages: [pymongo]
  extras_suggestion: null
  managed_free_tier: true       # M0 512 MB permanent
  license: SSPLv1 (server); Atlas is a service
  embedded_mode: false
  hybrid_search: true           # $search + $vectorSearch
  filter_dialect: MongoDB Query Language
  bulk_export: mongodump
  docs:
    install: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-quick-start/
    python_client: https://pymongo.readthedocs.io/
    pricing: https://www.mongodb.com/pricing
  verify_command: "python -c 'from pymongo import MongoClient; print(MongoClient().admin.command(\"ping\"))'"
  notes: "Flex GA Feb 2025: $8/mo base, capped at $30/mo."

- name: turbopuffer
  display_name: turbopuffer
  deployment: [managed]
  requires_server: false
  pip_packages: [turbopuffer]
  extras_suggestion: null
  managed_free_tier: false
  license: Proprietary
  embedded_mode: false
  hybrid_search: true
  filter_dialect: typed attribute filters
  bulk_export: list rows in pages
  docs:
    install: https://pypi.org/project/turbopuffer/
    python_client: https://github.com/turbopuffer/turbopuffer-python
    pricing: https://turbopuffer.com/pricing
  verify_command: "python -c 'from turbopuffer import Turbopuffer; print(Turbopuffer(region=\"gcp-us-central1\").namespaces.list())'"
  notes: "Stable v2.1.0 (May 17 2026). Region-pinned."

- name: pgvector
  display_name: pgvector (Postgres extension)
  deployment: [server, embedded-with-postgres]
  requires_server: true
  pip_packages: [pgvector, psycopg]
  extras_suggestion: "[binary]"
  managed_free_tier: false      # depends on PG host
  license: PostgreSQL-permissive
  embedded_mode: false
  hybrid_search: false          # manual SQL hybrid
  filter_dialect: SQL WHERE
  bulk_export: pg_dump
  docs:
    install: https://github.com/pgvector/pgvector
    python_client: https://github.com/pgvector/pgvector-python
    pricing: null
  verify_command: "psql -c \"SELECT extversion FROM pg_extension WHERE extname='vector'\""
  notes: "v0.8.2 (Feb 26 2026) fixes CVE-2026-3172 in parallel HNSW builds — upgrade."
```

### §7. Migration & vendor lock-in

Pick the portable interchange format first, then commit. **JSONL of `{id, vector, text, metadata}`** is the universally accepted lowest common denominator. **Parquet with an Arrow `FixedSizeList<Float32, dim>` vector column** is roughly 5–10× smaller and 10× faster to read. **`.npy`** is fine for vectors-only. Avoid pickle.

What is mechanical when you switch backends: re-upsert vectors. What is *not*: (1) **filter language rewrites** — every backend has its own dialect (Mongo-ish for Pinecone/Chroma, payload filters for Qdrant, GraphQL/`Filter` for Weaviate, SQL `WHERE` for pgvector/LanceDB, ES DSL for Elasticsearch, MongoDB Query for Atlas); a facade can abstract the common subset (`$eq`, `$in`, `$gt`, AND/OR) but not advanced features (Qdrant geo filters, Elastic nested fields, Weaviate references). (2) **Hybrid-search rewrite** — every backend handles BM25-vs-vector differently (RRF in ES, Hybrid in Weaviate/Redis, sparse+dense in Pinecone/Qdrant). (3) **ID type** — Pinecone is strings, Qdrant is UUIDs or unsigned ints, Weaviate requires UUID v4. (4) **Index re-tuning** — HNSW `M`, `efConstruction`, `efSearch` rarely transfer 1:1. **Effectively irreversible once you have data loaded:** vector dimension, distance metric, ID type. A facade should ask the user to pick these up front and store them in the registry.

### §8. Embedding integration — does the database embed for you?

**Document-first** (give it text, it embeds): Chroma (default `all-MiniLM-L6-v2`, swappable), Weaviate (vectorizer modules: `text2vec-openai`, `text2vec-cohere`, `text2vec-transformers`, `text2vec-ollama`), Pinecone (integrated inference with `create_index_for_model`), MongoDB Atlas (Auto Embeddings via Voyage AI), Vertex AI Vector Search 2.0 (auto-embeddings), Azure AI Search (integrated vectorization via Azure OpenAI), Marqo, Vespa (rank profiles + ONNX). **Vector-first** (you supply vectors): Qdrant (FastEmbed is *optional*), Milvus, Redis, Elasticsearch (unless you upload a model via Eland), FAISS, sqlite-vec, DuckDB-VSS, LanceDB, pgvector, Turbopuffer, VectorChord, Vald. **Facade design implication:** treat embedding as **external and optional**. Never assume the backend does it. Embedding-model choice is Report 02's problem; the facade just routes a `texts → vectors` callable (or a `texts → server-side-embed` shortcut where supported).

### §9. Offline / on-prem / special cases

**Air-gapped (zero network, zero telemetry by default):** FAISS, sqlite-vec, DuckDB-VSS, LanceDB (local mode), pgvector, VectorChord, Qdrant (self-host), Milvus standalone, Weaviate (self-host), Redis (self-host), Elasticsearch (self-host), Vald (k8s on-prem). Chroma phones home to PostHog by default — turn it off with `ANONYMIZED_TELEMETRY=False`. Weaviate also has telemetry — `DISABLE_TELEMETRY=true`. Pinecone, Turbopuffer, Vertex AI, Azure AI Search, MongoDB Atlas — all cloud-only, not viable air-gapped. **Object-storage-first / serverless economics** (compute and storage decoupled, cold tiers on S3/GCS): Turbopuffer (whole product), LanceDB-on-S3 (set `LANCE_*` env vars), Pinecone Serverless (no pod provisioning). Storage costs collapse, but cold queries pay a one-time fetch latency. **SQL-database-already-in-place path** ("you may not need a new database at all"): if you run **Postgres**, install pgvector (or VectorChord at >10M); **SQLite**, install sqlite-vec; **Redis**, use Redis 8 native vector; **Elasticsearch/OpenSearch**, use kNN + RRF; **DuckDB**, install VSS; **MongoDB**, use Atlas Vector Search ($vectorSearch).

### §10. Synthesis for the `vd` facade

**Recommended first six backends to credibly claim "switch vectorDBs freely", in implementation order:**

1. **Chroma** — easiest dev story, embedded + server + cloud parity, Mongo-ish filter (a natural common-subset dialect for the facade), Apache-2.0. This is the "hello world" backend.
2. **Qdrant** — single binary works embedded (in-client local mode), Docker, and managed; rich payload filters; clean Python SDK; permanent free cloud. Cleanest mapping to a facade's VectorStore Protocol.
3. **pgvector** — covers "I already run Postgres", which is the most common situation. The relational join + vector query is a unique escape hatch worth documenting.
4. **LanceDB** — embedded multimodal, S3-native, schema evolution. Demonstrates that the facade is not just a thin wrapper around three lookalikes.
5. **Pinecone** — managed-first canonical. Forces the facade to handle "the DB embeds for me" (integrated inference) and "no native bulk export" cases.
6. **FAISS** — pure in-memory benchmark / baseline. Demonstrates the brute-force escape hatch and gives the facade a recall@k oracle.

**`check_requirements(backend)` should detect, in priority order, per archetype:**

- *Embedded:* Python version (sqlite-vec ≥3.41 needs SQLite check; Milvus Lite needs Linux/macOS, not Windows; FAISS — CPU SIMD detection); pip package installed; for sqlite-vec, that `sqlite3.Connection.enable_load_extension` exists; for DuckDB-VSS, that the `vss` extension installs.
- *Self-hosted server:* Docker daemon up; expected port free (6333, 8080, 19530, 6379, 9200…); container running and healthy (curl the health endpoint); client library major version compatible with server major version.
- *Managed cloud:* required env var present (`PINECONE_API_KEY`, `WEAVIATE_API_KEY` + `WEAVIATE_URL`, `QDRANT_API_KEY` + `QDRANT_URL`, `TURBOPUFFER_API_KEY`, `MONGODB_URI`); network reachability to the API endpoint; minimal "list" call succeeds.

Print the *next* step on failure: "Run `docker run -p 6333:6333 qdrant/qdrant`" or "Set `PINECONE_API_KEY` from app.pinecone.io" or "`pip install -U weaviate-client`".

**Hardcode in registry vs re-verify at runtime.** Hardcode: pip package names, deployment archetypes, license, embedded-mode bool, default ports, filter-dialect family, hybrid-search capability bool, bulk-export mechanism, ID type expected, env var names. Re-verify at runtime (these drift on the order of months — link to the official pricing page instead of caching the number): pricing tiers, free-tier limits, latest stable version, default index type (e.g. Elasticsearch's default switched to `int8_hnsw` then to `bbq_hnsw`), license text (Redis just changed, Marqo just deprecated OSS, Elastic just added AGPLv3). The registry should store *URLs to pricing/limits/docs*, not *the numbers*.

## Recommendations

**For an end user picking a backend right now (May 2026):**

1. **Start at §4's decision tree.** Answer Q1–Q6 honestly. The first matching row in the rubric table is your default. Move to the runner-up only if a hard blocker hits (license incompatible with your org; free tier insufficient; team can't run Docker).
2. **If you're under 100k vectors and one machine, do not pick a vector DB yet.** Build with `sklearn.neighbors.NearestNeighbors` or `faiss.IndexFlatIP`. Revisit when persistence, multi-process, or filtering at >100k forces your hand.
3. **If you already run Postgres, install pgvector 0.8.2 today.** Do not add a second database for vector search until you have a measured reason. Upgrade to ≥0.8.2 if you do parallel HNSW builds (CVE-2026-3172). Cross over to VectorChord at >10M vectors.
4. **For a managed, zero-ops choice, default to Qdrant Cloud free tier first, Pinecone Serverless second.** Qdrant has the cleanest exit path; Pinecone has the most polished UX and namespace model. Avoid Weaviate Cloud for indefinite dev — the sandbox is 14 days only.
5. **For embedded, default to Chroma; pick LanceDB instead if you need multimodal, S3, or schema evolution; pick sqlite-vec if your app already uses SQLite.** Pick FAISS only when you need a pure in-memory benchmark or a library, not a database.
6. **Run your own benchmark before committing at scale.** Three numbers on your real corpus: recall@10 vs. brute force, p95 query latency at your target QPS, and projected $/month at 10× your current size. If a backend fails any of these, switch *before* you have 10M vectors in it.

**For the `vd` facade maintainer:**

- Ship the six backends in §10's order. Stop after the first two if budget is tight — Chroma + Qdrant covers ≥70% of users.
- Build `check_requirements(backend)` exactly as scoped in §10. The "what next" string is the highest-leverage UX.
- Hardcode only the structural facts. Cache *URLs*, not *prices*. The TODO/refresh date in the registry should be visible.
- Default the facade's filter dialect to the Mongo-ish subset (`$eq`, `$in`, `$and`, `$gt`, `$lt`). Document an "escape hatch" per backend for native-dialect passthrough (Weaviate `Filter`, Qdrant payload filter, ES query DSL, SQL `WHERE`).
- Treat embedding as injected. Never call out to OpenAI / Ollama from inside the facade — the user passes a `encode(texts) → vectors` callable, or opts into the backend's integrated inference where it exists.

**Thresholds that would change the recommendation:**

- Pinecone Starter tier limits change (currently 5 indexes / 2 GB / AWS us-east-1): re-evaluate against Qdrant Cloud free.
- Weaviate restores a permanent free managed tier: it re-enters the "default managed pick" set.
- pgvector remains <100ms p95 at your scale: don't add a dedicated vector DB.
- A backend's hosted region matters for residency (EU/UK/SG/IN): Turbopuffer and Qdrant Cloud advertise the most regions, Pinecone serverless is more limited.

## Caveats

- **Pricing and free-tier limits are the most volatile facts in this report.** Treat every number as a pointer to the linked pricing page, not as a contract. Re-verify before quoting in code or a proposal.
- **License changes are accelerating.** In the 24 months before this report: Redis (BSD → SSPL/RSAL → AGPLv3), Elastic (Apache → SSPL/ELv2 → AGPLv3 added), pgvecto.rs (deprecated), Marqo OSS (deprecated). Always check the current `LICENSE` file before assuming open-source status. If you build a SaaS on top of an AGPLv3 backend, get legal advice on whether your application code is implicated.
- **Specific package versions drift.** The pinecone-client → pinecone rename caught many older tutorials; the `weaviate-client` v3 → v4 typed-API shift broke older code; `pymilvus` + Milvus Lite is being rewritten from C++/Cgo to pure-Python with an incompatible `.db` format. Always pin majors and verify against the official quickstart linked in §5.
- **The "managed" backends and self-hostable backends are not equivalent.** Pinecone has no documented self-host path (BYOC ≠ self-host); Turbopuffer is managed-only; Vertex AI / Azure AI Search are cloud-only. If portability or air-gapped operation is non-negotiable, restrict your choices to the Apache-2.0 / BSD-3 / AGPLv3 self-hostable set.
- **A few numbers in §5 come from official quickstarts that may have been revised after May 2026.** Notably, Pinecone Starter limits, Weaviate Cloud tier pricing, MongoDB Atlas Flex cap, and pgvector's CVE-fix release are all dated explicitly — re-verify via the linked official URL before depending on them.
- **Report 03 [1] remains the source of truth for ANN algorithm theory** (HNSW vs IVF vs PQ vs DiskANN vs ScaNN), the facade-design contract, the filter-language comparison, and the hybrid-search abstraction. This report deliberately defers to it on those questions and only mentions index types when they affect setup.

---

## REFERENCES

[1] [Report 03 — Vector Storage and Retrieval (companion theory report, April 2026)](https://docs/research/semantic_search/03)
[2] [pinecone-io/pinecone-python-client (GitHub)](https://github.com/pinecone-io/pinecone-python-client)
[3] [Pinecone Python SDK documentation](https://sdk.pinecone.io/python/index.html)
[4] [Pinecone pricing](https://www.pinecone.io/pricing/)
[5] [Pinecone Database API limits](https://docs.pinecone.io/reference/api/database-limits)
[6] [Qdrant Cloud pricing](https://qdrant.tech/pricing/)
[7] [Qdrant Local Quickstart](https://qdrant.tech/documentation/quickstart/)
[8] [Qdrant Python client documentation](https://python-client.qdrant.tech/)
[9] [Qdrant Docker Hub image](https://hub.docker.com/r/qdrant/qdrant)
[10] [Weaviate pricing page](https://weaviate.io/pricing)
[11] [Weaviate Cloud pricing update blog (Oct 27, 2025)](https://weaviate.io/blog/weaviate-cloud-pricing-update)
[12] [Weaviate Python client v4 GA blog](https://weaviate.io/blog/py-client-v4-release)
[13] [Weaviate Python client documentation](https://docs.weaviate.io/weaviate/client-libraries/python)
[14] [Weaviate Cloud Quickstart](https://docs.weaviate.io/cloud/quickstart)
[15] [weaviate-client on PyPI](https://pypi.org/project/weaviate-client/)
[16] [Milvus standalone Docker install](https://milvus.io/docs/install_standalone-docker.md)
[17] [Milvus Docker Compose install](https://milvus.io/docs/install_standalone-docker-compose.md)
[18] [Milvus Lite documentation](https://milvus.io/docs/milvus_lite.md)
[19] [chromadb on PyPI](https://pypi.org/project/chromadb/)
[20] [Chroma homepage and product page](https://www.trychroma.com/products/chromadb)
[21] [LanceDB Quickstart](https://docs.lancedb.com/quickstart)
[22] [lancedb on PyPI](https://pypi.org/project/lancedb/)
[23] [LanceDB SDK Reference](https://lancedb.github.io/lancedb/)
[24] [sqlite-vec installation](https://alexgarcia.xyz/sqlite-vec/installation.html)
[25] [sqlite-vec Python guide](https://alexgarcia.xyz/sqlite-vec/python.html)
[26] [sqlite-vec on PyPI](https://pypi.org/project/sqlite-vec/)
[27] [sqlite-vec GitHub](https://github.com/asg017/sqlite-vec)
[28] [DuckDB VSS extension documentation](https://duckdb.org/docs/current/core_extensions/vss)
[29] [DuckDB VSS announcement blog](https://duckdb.org/2024/05/03/vector-similarity-search-vss)
[30] [pgvector GitHub (install, CVE-2026-3172 fix in v0.8.2 — Feb 26, 2026)](https://github.com/pgvector/pgvector)
[31] [pgvector Python bindings on PyPI](https://pypi.org/project/pgvector/)
[32] [pgvector-python GitHub](https://github.com/pgvector/pgvector-python)
[33] [Redis vector search concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
[34] [Redis vector database quickstart](https://redis.io/docs/latest/develop/get-started/vector-database/)
[35] [RediSearch GitHub](https://github.com/RediSearch/RediSearch)
[36] [Redis Vector Library (RedisVL)](https://github.com/redis/redis-vl-python)
[37] [Elasticsearch dense_vector reference](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector)
[38] [Elasticsearch kNN search](https://www.elastic.co/docs/solutions/search/vector/knn)
[39] [Elastic licensing FAQ (AGPLv3 added Aug 2024)](https://www.elastic.co/pricing/faq/licensing)
[40] [MongoDB Atlas pricing](https://www.mongodb.com/pricing)
[41] [MongoDB Atlas Flex tier GA blog (Feb 6, 2025 — $30/mo cap)](https://www.mongodb.com/company/blog/product-release-announcements/dynamic-workloads-predictable-costs-mongodb-atlas-flex-tier)
[42] [Turbopuffer pricing](https://turbopuffer.com/pricing)
[43] [Turbopuffer documentation](https://turbopuffer.com/docs)
[44] [turbopuffer-python GitHub (v2.1.0, May 17, 2026)](https://github.com/turbopuffer/turbopuffer-python)
[45] [faiss-cpu on PyPI (v1.13.2, Dec 24, 2025)](https://pypi.org/project/faiss-cpu/)
[46] [FAISS INSTALL.md](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
[47] [Vespa Quick Start](https://docs.vespa.ai/en/vespa-quick-start.html)
[48] [Vespa Cloud](https://cloud.vespa.ai/)
[49] [VectorChord (tensorchord/VectorChord) GitHub](https://github.com/tensorchord/VectorChord)
[50] [pgvecto.rs GitHub — deprecated, points to VectorChord](https://github.com/tensorchord/pgvecto.rs)
[51] [Vald project homepage](https://vald.vdaas.org/)
[52] [Marqo GitHub — OSS deprecation notice](https://github.com/marqo-ai/marqo)
[53] [Vertex AI Vector Search overview](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
[54] [Azure AI Search vector search overview](https://learn.microsoft.com/en-us/azure/search/vector-search-overview)
[55] [Redis 8 returns to open source under AGPLv3 (InfoQ, May 2025)](https://www.infoq.com/news/2025/05/redis-agpl-license/)
[56] [Elastic announces open source license for Elasticsearch and Kibana (Aug 2024)](https://ir.elastic.co/news/news-details/2024/Elastic-Announces-Open-Source-License-for-Elasticsearch-and-Kibana-Source-Code/default.aspx)