---
name: vd-backend-choose
description: >-
  Backend-selection tooling for the vd package. Use this skill when the user is
  picking a vector database with vd, hits a "Backend X is not available" error,
  asks "which backend should I use", weighs persistence / cloud / cost / speed
  trade-offs, or wants to drive the connection from a YAML/TOML config file
  with profiles and environment variables.
audience: users
---

# Choosing and connecting to a vd backend

The `vd` package ships with `memory` (always available) and `chroma` (optional)
and has stubs for several planned backends (`pinecone`, `weaviate`, `qdrant`,
`milvus`, `faiss`). This skill covers: discovering what's available, getting
install instructions, choosing one for the user's needs, and driving the
connection from a config file.

For the basic happy path once a backend is chosen, see `vd-quickstart`.

## Discovering backends

```python
import vd

vd.list_backends()              # ['memory', 'chroma']  — registered classes
vd.list_available_backends()    # ['memory']            — actually importable
vd.list_all_backends(include_planned=True)  # dict with metadata + status
vd.print_backends_table()       # pretty grouped output to stdout
vd.get_backend_info('chroma')   # full metadata dict for one backend
```

Three subtly different lists:

- **`list_backends()`** — backends *registered with `@register_backend`* in the
  current process. Always includes `memory`.
- **`list_available_backends()`** — backends whose `module_check` import
  succeeds (i.e., the optional dep is installed).
- **`list_all_backends()`** — every known backend in `vd._backend_metadata`,
  with `available` and `registered` flags added. Pass
  `include_planned=True` to include not-yet-implemented ones.

Use `list_available_backends()` when you need to know what works *right now*
without installing anything.

## Handling "not available"

```python
>>> vd.connect('chroma')
ValueError: Backend 'chroma' is not available.

To install it:
  pip install vd[chromadb]

Or run: vd.get_install_instructions('chroma') for more details.
```

When this happens, do **not** silently fall back to `memory` unless the user
asked for that. Show them the install instructions:

```python
print(vd.get_install_instructions('chroma'))
```

`get_install_instructions(name)` returns a multi-line string with description,
features, limitations, and the pip install command. If the backend is
`'planned'` it says so explicitly.

## Recommending a backend

```python
vd.print_recommendation(
    dataset_size='medium',         # 'small' | 'medium' | 'large' | 'very_large'
    persistence_required=True,
    cloud_required=False,
    budget='free',                 # 'free' | 'low' | 'medium' | 'high'
    performance_priority='balanced',  # 'speed' | 'scalability' | 'balanced'
)

# Programmatic version:
rec = vd.recommend_backend(
    dataset_size='small',
    persistence_required=True,
    budget='free',
)
# rec['recommended'] -> ranked list; rec['reasoning'] -> per-backend notes
```

Heuristics it uses:

- `dataset_size='small'` (≤ ~10k docs) and `persistence_required=False`
  → `memory` is fine.
- `persistence_required=True` and `budget='free'` → `chroma`.
- `cloud_required=True` or `dataset_size='very_large'` → `pinecone` / `milvus`
  (planned — surface to the user that they're not yet implemented).

For a side-by-side feature comparison without the prescriptive ranking:

```python
vd.print_comparison(['memory', 'chroma', 'pinecone'])
chars = vd.get_backend_characteristics()  # dict of features per backend
```

## Quick decision rubric

When the user is undecided, default to:

| Use case | Backend |
|---|---|
| Throwaway prototype, REPL exploration, unit tests | `memory` |
| Local app that should remember across runs | `chroma(persist_directory=...)` |
| Production, multi-machine, or > millions of vectors | `pinecone` / `weaviate` / `qdrant` (currently planned — flag this) |

If the user asks for a planned backend, tell them it's not yet implemented in
`vd` and offer either (a) `chroma` as a stand-in or (b) using the underlying
client (e.g. `pinecone-client`) directly.

## Connecting from a config file

`vd` supports YAML and TOML config files with **profiles** (named groups of
settings) and **env-var overrides**. This is the right path when the same code
must run against `memory` in tests and `chroma` (or future backends) in prod.

```python
import vd

# Use the default profile from ./vd.yaml (or vd.toml, etc.)
client = vd.connect_from_config('vd.yaml')

# Pick a specific profile
client = vd.connect_from_config('vd.yaml', profile='production')

# Override profile values inline
client = vd.connect_from_config(
    'vd.yaml',
    profile='production',
    persist_directory='./alt_dir',  # forwarded as backend kwarg
)
```

Example `vd.yaml`:

```yaml
profiles:
  default:
    backend: memory
  dev:
    backend: memory
  prod:
    backend: chroma
    persist_directory: ./vector_db
    embedding_model: text-embedding-3-small
```

Environment variables (applied when `apply_env=True`, the default):

- `VD_PROFILE` — selects which profile to load (default: `default`)
- `VD_BACKEND` — overrides the `backend` field
- `VD_EMBEDDING_MODEL` — overrides the `embedding_model` field

Generate a starter file:

```python
import vd
yaml_text = vd.create_example_config('yaml')   # or 'toml'
vd.save_config({'profiles': {'default': {'backend': 'memory'}}}, 'vd.yaml')
```

Lower-level helpers if you need them:

- `vd.load_config(path, *, format=None)` — returns the raw dict
- `vd.save_config(config, path, *, format=None)` — writes a dict to file
- `vd.create_example_config(format='yaml')` — string with example content

## Common gotchas

- **`memory` is registered but not persistent.** Restarting the process loses
  everything. Don't recommend `memory` for any flow that runs more than once.
- **Planned backends are in `_backend_metadata` but not in `_backends`.**
  `vd.list_backends()` will not show them; `vd.list_all_backends(include_planned=True)`
  will. `vd.connect('pinecone')` raises a `ValueError` with a "planned"
  message — don't pretend it works.
- **Config files are optional.** If the user only runs in one environment,
  `vd.connect('memory')` is simpler. Reach for `connect_from_config` when
  there's a real reason (multi-env, secrets in env vars, ops handoff).
- **`embedding_model` in YAML is a string only** (e.g. `'text-embedding-3-small'`).
  To pass a callable embedder, use `vd.connect_from_config(..., embedding_model=fn)`
  — that kwarg overrides whatever the file says.
- **Don't shadow `connect` with `connect_from_config` automatically.** They
  are different entry points; pick one based on whether the user has (or wants)
  a config file.
