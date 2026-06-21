# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/);
each section corresponds to a git version tag (which is also the release
published to PyPI). Entries are commit subjects and PR titles, verbatim.

## [0.2.9] - 2026-06-06

### Added

- feat(search): BM25Index — build-once / query-many lexical index ([#23](https://github.com/i2mint/vd/pull/23))

## [0.2.8] - 2026-05-28

### Docs

- docs(base): canonical SearchResult score contract; align faiss l2 ([#21](https://github.com/i2mint/vd/pull/21))

## [0.2.7] - 2026-05-27

- Add AsyncClient + AsyncCollection — Phase 1 (universal wrapper) ([#19](https://github.com/i2mint/vd/pull/19))

## [0.2.6] - 2026-05-27

- Add SupportsHybrid + client-side RRF fallback ([#16](https://github.com/i2mint/vd/pull/16))

## [0.2.5] - 2026-05-27

- ci: refresh stub for new permissions block

## [0.2.4] - 2026-05-24

- ci: refresh stub for new permissions block

## [0.2.3] - 2026-05-24

- ci: switch to wads reusable workflow stub

## [0.2.2] - 2026-05-22

- Live-verify server/managed backend adapters; fix 8 bugs found ([#14](https://github.com/i2mint/vd/pull/14))

## [0.2.1] - 2026-05-21

- v0.2: vector-first facade redesign + full backend build-out (15 backends) ([#12](https://github.com/i2mint/vd/pull/12))

## [0.1.6] - 2026-05-20

- Harden the vd facade contract: filter language, capability protocols, escape hatch ([#6](https://github.com/i2mint/vd/pull/6))

## [0.1.5] - 2026-05-16

- Add TimeIndexedCollection — time-windowed wrapper over any Collection ([#3](https://github.com/i2mint/vd/pull/3))

## [0.1.4] - 2026-05-14

- chore(ci): bump action pins to checkout@v6, setup-uv@v7

## [0.1.3] - 2026-04-27

- chore: move stuff around

## [0.1.2] - 2026-04-27

### Added

- feat: Add bundled AI-agent skills and restructure README around three interfaces

## [0.1.1] - 2026-04-27

- chore: Migrate CI to uv-based workflow
- style: Simplify formatting in pyproject.toml for consistency
- Incorporate changes from claude/implement-package-01KNcp8ZV9pnHshUo2xqLLBM
- Refactor code structure for improved readability and maintainability
- pyproject
- specifications

### Added

- feat: Add initial docstring for VectorDB facades

## [0.0.11] - 2025-08-22

- edit setup.cfg

## [0.0.10] - 2025-07-09

### Added

- feat: Enhance prepare_for_crude_dispatch to handle dictionary store_keys

## [0.0.9] - 2025-07-01

### Added

- feat: Add integration test for source_variables decorator with FastAPI

## [0.0.8] - 2025-07-01

- Update ci.yml
- [options.extras_require]  pytest-asyncio
- 0.0.7:
- Implement comprehensive tests for DOG and ADOG operations with local data stores
- Implement CRUDE (CRUD-Execution) framework for managing complex Python objects through string keys and stores. Introduce functions for automatic key generation, output storage, and dispatching with support for both synchronous and asynchronous functions. Add utility classes and functions for easier CRUD operations, including a mall structure for organizing stores. Enhance flexibility with auto-naming and output transformation capabilities.
- 0.0.6:
- 0.0.3:

### Added

- feat: Implement input wiring functionality with dependency injection and add comprehensive test suite
- feat: Enhance test execution with command-line arguments for store type selection
- feat: Add DOG and ADOG classes for synchronous and asynchronous data operation execution
- feat: Implement Data Operation Graph (DOG) and Asynchronous DOG (ADOG) classes

### Changed

- refactor: kwargs_from_args_and_kwargs -> map_arguments & args_and_kwargs_from_kwargs -> mk_args_and_kwargs
