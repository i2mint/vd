# vd.requirements

Setup assistance: turning provider metadata into actionable diagnostics.

Choosing a vector database ([`vd.providers`](vd.html.md#vd.providers)) is half the job; \*getting it
running\* is the other half, and the effort is dominated by the deployment
archetype, not the algorithm. This module implements the report’s §10
`check_requirements` scope:

- **embedded** backends — is the pip package importable? platform/version
  quirks (Milvus Lite is not native-Windows; sqlite-vec needs SQLite >= 3.41)?
- **server** backends — is the client installed, and is something answering on
  the expected port?
- **managed** backends — is the client installed, and are the required
  environment variables set?

Every check ends with the single highest-leverage output: the *next step* —
the exact command or action that moves the user forward.

### Functions

| [`check_requirements`](#vd.requirements.check_requirements)(backend, \*[, verbose])   | Diagnose whether `backend` is ready to use, and say what to do if not.   |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`install_backend`](#vd.requirements.install_backend)(backend, \*[, run])          | Return (and optionally run) the `pip install` command for `backend`.     |
| [`setup_guide`](#vd.requirements.setup_guide)(backend)                         | Return a full, copy-pasteable setup playbook for `backend`.              |

### vd.requirements.check_requirements(backend, , verbose=True)

Diagnose whether `backend` is ready to use, and say what to do if not.

Runs an installed-check plus archetype-specific checks (embedded / server /
managed), then computes the single most useful *next step*.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A provider name (see [`vd.list_all_backends()`](vd.html.md#vd.list_all_backends)).
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print a human-readable report (in addition to returning the dict).
* **Returns:**
  `{"backend", "archetype", "ok", "checks", "next_step"}` where
  `checks` is a list of `{"name", "ok", "detail"}` records.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> report = check_requirements('memory', verbose=False)
>>> report['ok']
True
```

### vd.requirements.install_backend(backend, , run=False)

Return (and optionally run) the `pip install` command for `backend`.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Provider name.
  * **run** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `True`, actually invoke pip in the current interpreter. If
    `False` (the default), only return the command — the caller decides.
* **Returns:**
  The pip command (or a note that nothing is needed).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### vd.requirements.setup_guide(backend)

Return a full, copy-pasteable setup playbook for `backend`.

Covers: the pip install, a Docker one-liner for server backends, the
environment variables for managed backends, a verify command, and the
relevant documentation links.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
