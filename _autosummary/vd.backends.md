# vd.backends

Backend adapters for the vector databases `vd` supports.

Importing this package registers every adapter whose client library is
installed. Each adapter lives in its own module and registers itself with the
[`register_backend()`](vd.util.md#vd.util.register_backend) decorator; modules whose third-party client
is not installed fail to import and are skipped silently — that backend simply
will not appear in [`vd.list_backends()`](vd.md#vd.list_backends).

### Modules

| [`memory`](vd.backends.memory.md#module-vd.backends.memory)   | In-memory backend — the reference adapter.   |
|-------------------------------------------------------------------------------------|----------------------------------------------|
