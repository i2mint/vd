"""
Setup assistance: turning provider metadata into actionable diagnostics.

Choosing a vector database (:mod:`vd.providers`) is half the job; *getting it
running* is the other half, and the effort is dominated by the deployment
archetype, not the algorithm. This module implements the report's §10
``check_requirements`` scope:

- **embedded** backends — is the pip package importable? platform/version
  quirks (Milvus Lite is not native-Windows; sqlite-vec needs SQLite >= 3.41)?
- **server** backends — is the client installed, and is something answering on
  the expected port?
- **managed** backends — is the client installed, and are the required
  environment variables set?

Every check ends with the single highest-leverage output: the *next step* —
the exact command or action that moves the user forward.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sqlite3
import sys
from typing import Any, Optional

from vd.providers import install_command, provider

# --------------------------------------------------------------------------- #
# One-liner server bring-up commands (report §5). Structural — change yearly.
# --------------------------------------------------------------------------- #

_DOCKER_COMMANDS: dict[str, str] = {
    "qdrant": (
        "docker run -p 6333:6333 -p 6334:6334 "
        "-v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant"
    ),
    "weaviate": (
        "docker run -p 8080:8080 -p 50051:50051 "
        "-e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true "
        "cr.weaviate.io/semitechnologies/weaviate:latest"
    ),
    "milvus": (
        "curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/"
        "scripts/standalone_embed.sh -o standalone_embed.sh "
        "&& bash standalone_embed.sh start"
    ),
    "redis": "docker run -p 6379:6379 redis:8",
    "elasticsearch": (
        "docker run -p 9200:9200 -e discovery.type=single-node "
        "-e xpack.security.enabled=false "
        "docker.elastic.co/elasticsearch/elasticsearch:8.18.0"
    ),
    "opensearch": (
        "docker run -p 9200:9200 -e discovery.type=single-node "
        "-e plugins.security.disabled=true "
        "opensearchproject/opensearch:latest"
    ),
    "pgvector": "docker run -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg17",
    "vespa": "docker run -p 8080:8080 -p 19071:19071 vespaengine/vespa",
}


# --------------------------------------------------------------------------- #
# Low-level checks
# --------------------------------------------------------------------------- #


def _check(
    name: str, ok: bool, detail: str, *, optional: bool = False
) -> dict[str, Any]:
    """
    Build one check record.

    An ``optional`` check is reported but does not gate overall readiness — used
    for things like "a server is running" on a backend that also runs embedded.
    """
    return {"name": name, "ok": ok, "detail": detail, "optional": optional}


def _port_open(host: str, port: int, *, timeout: float = 1.0) -> bool:
    """Return whether a TCP connection to ``host:port`` succeeds quickly."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _module_importable(pip_package: str) -> bool:
    """Return whether the module a pip package provides can be found."""
    from vd.providers import _import_name

    try:
        return importlib.util.find_spec(_import_name(pip_package)) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Archetype-specific requirement checks
# --------------------------------------------------------------------------- #


def _embedded_checks(name: str, meta: dict) -> list[dict[str, Any]]:
    """Checks for an embedded / library backend."""
    checks: list[dict[str, Any]] = []

    if name == "sqlite_vec":
        major_minor = tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:2])
        checks.append(
            _check(
                "sqlite version",
                major_minor >= (3, 41),
                f"SQLite {sqlite3.sqlite_version} (sqlite-vec wants >= 3.41)",
            )
        )
        probe = sqlite3.connect(":memory:")
        can_load = hasattr(probe, "enable_load_extension")
        probe.close()
        checks.append(
            _check(
                "extension loading",
                can_load,
                "sqlite3 supports enable_load_extension"
                if can_load
                else "this Python's sqlite3 has extension loading disabled "
                "(use Homebrew Python or pysqlite3-binary)",
            )
        )

    if name == "milvus" and sys.platform.startswith("win"):
        checks.append(
            _check(
                "platform",
                False,
                "Milvus Lite is not available on native Windows — use WSL2",
            )
        )

    return checks


def _server_checks(name: str, meta: dict) -> list[dict[str, Any]]:
    """
    Checks for a self-hosted server backend.

    When the backend also has an embedded mode (Qdrant, Milvus, Weaviate), the
    server-reachable check is *optional*: ``vd.connect`` works embedded without
    any server, so a missing server should not report the backend "not ready".
    """
    port = meta.get("default_port")
    if not port:
        return []
    reachable = _port_open("localhost", int(port))
    embedded = bool(meta.get("embedded_mode"))
    if reachable:
        detail = f"a server is answering on localhost:{port}"
    elif embedded:
        detail = (
            f"no server on localhost:{port} — fine for embedded mode; "
            f"start one only for server mode"
        )
    else:
        detail = f"nothing is answering on localhost:{port}"
    return [_check("server reachable", reachable, detail, optional=embedded)]


def _managed_checks(name: str, meta: dict) -> list[dict[str, Any]]:
    """Checks for a managed cloud backend."""
    checks = []
    for var in meta.get("env_vars") or []:
        present = bool(os.environ.get(var))
        checks.append(
            _check(
                f"env var {var}",
                present,
                "set" if present else f"{var} is not set",
            )
        )
    return checks


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #


def check_requirements(backend: str, *, verbose: bool = True) -> dict[str, Any]:
    """
    Diagnose whether ``backend`` is ready to use, and say what to do if not.

    Runs an installed-check plus archetype-specific checks (embedded / server /
    managed), then computes the single most useful *next step*.

    Parameters
    ----------
    backend : str
        A provider name (see :func:`vd.list_all_backends`).
    verbose : bool
        Print a human-readable report (in addition to returning the dict).

    Returns
    -------
    dict
        ``{"backend", "archetype", "ok", "checks", "next_step"}`` where
        ``checks`` is a list of ``{"name", "ok", "detail"}`` records.

    Examples
    --------
    >>> report = check_requirements('memory', verbose=False)
    >>> report['ok']
    True
    """
    meta = provider(backend)
    if meta is None:
        result = {
            "backend": backend,
            "archetype": None,
            "ok": False,
            "checks": [_check("known backend", False, f"{backend!r} is unknown")],
            "next_step": "Run vd.print_backends_table() to see valid names.",
        }
        if verbose:
            _print_report(result)
        return result

    archetype = meta.get("archetype")
    checks: list[dict[str, Any]] = []

    # 1. Client library installed?
    packages = meta.get("pip_packages") or []
    installed = (not packages) or _module_importable(packages[0])
    checks.append(
        _check(
            "client installed",
            installed,
            "client library is importable"
            if installed
            else f"client library is missing ({install_command(backend)})",
        )
    )

    # 2. Archetype-specific checks (only meaningful once installed).
    if installed:
        if archetype in ("embedded", "library"):
            checks += _embedded_checks(backend, meta)
        elif archetype == "server":
            checks += _server_checks(backend, meta)
        elif archetype == "managed":
            checks += _managed_checks(backend, meta)

    # Optional checks are reported but do not gate overall readiness.
    ok = all(c["ok"] for c in checks if not c.get("optional"))
    next_step = _next_step(backend, meta, checks, ok)

    result = {
        "backend": backend,
        "archetype": archetype,
        "ok": ok,
        "checks": checks,
        "next_step": next_step,
    }
    if verbose:
        _print_report(result)
    return result


def _next_step(backend: str, meta: dict, checks: list[dict], ok: bool) -> str:
    """Compute the single highest-leverage next action."""
    if ok:
        return f"Ready. Connect with: vd.connect({backend!r}, ...)"

    by_name = {c["name"]: c for c in checks}

    if not by_name.get("client installed", {}).get("ok", True):
        return install_command(backend)

    if not by_name.get("server reachable", {}).get("ok", True):
        docker = _DOCKER_COMMANDS.get(backend)
        if docker:
            return f"Start the server:\n  {docker}"
        return f"Start the {meta['display_name']} server and retry."

    for c in checks:
        if c["name"].startswith("env var") and not c["ok"]:
            var = c["name"].removeprefix("env var ").strip()
            docs = (meta.get("docs") or {}).get("install", "")
            return f"Set the {var} environment variable" + (
                f" (see {docs})" if docs else ""
            )

    # Any other failing check — surface its detail.
    for c in checks:
        if not c["ok"]:
            return c["detail"]
    return "Unknown issue — see the checks above."


def _print_report(result: dict[str, Any]) -> None:
    """Print a :func:`check_requirements` result readably."""
    status = "READY" if result["ok"] else "NOT READY"
    print(f"\ncheck_requirements({result['backend']!r}) — {status}")
    print("=" * 60)
    for c in result["checks"]:
        mark = "OK  " if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    print()
    print(f"  NEXT STEP: {result['next_step']}")
    print()


# --------------------------------------------------------------------------- #
# Setup guide & optional installer
# --------------------------------------------------------------------------- #


def setup_guide(backend: str) -> str:
    """
    Return a full, copy-pasteable setup playbook for ``backend``.

    Covers: the pip install, a Docker one-liner for server backends, the
    environment variables for managed backends, a verify command, and the
    relevant documentation links.
    """
    meta = provider(backend)
    if meta is None:
        raise ValueError(f"Unknown backend {backend!r}")

    lines = [f"Setting up: {meta['display_name']}  ({backend})", "=" * 60]
    lines.append(f"Archetype: {meta.get('archetype')}   License: {meta.get('license')}")
    lines.append("")
    lines.append("1. Install the client:")
    lines.append(f"   {install_command(backend)}")

    if backend in _DOCKER_COMMANDS:
        lines += [
            "",
            "2. Start the server (Docker one-liner):",
            f"   {_DOCKER_COMMANDS[backend]}",
        ]

    env_vars = meta.get("env_vars") or []
    if env_vars and meta.get("archetype") == "managed":
        step = 2 if backend not in _DOCKER_COMMANDS else 3
        lines += ["", f"{step}. Set credentials (never commit these):"]
        for var in env_vars:
            lines.append(f"   export {var}=...")

    if meta.get("verify_command"):
        lines += ["", "Verify:", f"   {meta['verify_command']}"]

    docs = meta.get("docs") or {}
    if docs.get("install") or docs.get("python_client"):
        lines += ["", "Docs:"]
        for label in ("install", "python_client", "filtering", "pricing"):
            if docs.get(label):
                lines.append(f"   {label:14} {docs[label]}")

    if meta.get("notes"):
        lines += ["", f"Note: {meta['notes']}"]

    return "\n".join(lines)


def install_backend(backend: str, *, run: bool = False) -> str:
    """
    Return (and optionally run) the ``pip install`` command for ``backend``.

    Parameters
    ----------
    backend : str
        Provider name.
    run : bool
        If ``True``, actually invoke pip in the current interpreter. If
        ``False`` (the default), only return the command — the caller decides.

    Returns
    -------
    str
        The pip command (or a note that nothing is needed).
    """
    cmd = install_command(backend)
    if run and cmd.startswith("pip install "):
        import subprocess

        packages = cmd.removeprefix("pip install ").split()
        subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)
    return cmd
