# vd.cli

Command-line interface for vd.

Provides a CLI tool for common vd operations like listing backends,
exporting/importing collections, health checks, and more.

### Functions

| [`cmd_backends`](#vd.cli.cmd_backends)(args)     | List available backends.                     |
|-------------------------------------------------------------------------|----------------------------------------------|
| [`cmd_benchmark`](#vd.cli.cmd_benchmark)(args)    | Benchmark search performance.                |
| [`cmd_export`](#vd.cli.cmd_export)(args)       | Export a collection.                         |
| [`cmd_health_check`](#vd.cli.cmd_health_check)(args) | Check backend health.                        |
| [`cmd_import`](#vd.cli.cmd_import)(args)       | Import into a collection.                    |
| [`cmd_install_info`](#vd.cli.cmd_install_info)(args) | Get installation instructions for a backend. |
| [`cmd_migrate`](#vd.cli.cmd_migrate)(args)      | Migrate a collection between backends.       |
| [`cmd_stats`](#vd.cli.cmd_stats)(args)        | Show collection statistics.                  |
| [`cmd_validate`](#vd.cli.cmd_validate)(args)     | Validate a collection.                       |
| [`main`](#vd.cli.main)()                 | Main CLI entry point.                        |

### vd.cli.cmd_backends(args)

List available backends.

### vd.cli.cmd_benchmark(args)

Benchmark search performance.

### vd.cli.cmd_export(args)

Export a collection.

### vd.cli.cmd_health_check(args)

Check backend health.

### vd.cli.cmd_import(args)

Import into a collection.

### vd.cli.cmd_install_info(args)

Get installation instructions for a backend.

### vd.cli.cmd_migrate(args)

Migrate a collection between backends.

### vd.cli.cmd_stats(args)

Show collection statistics.

### vd.cli.cmd_validate(args)

Validate a collection.

### vd.cli.main()

Main CLI entry point.
