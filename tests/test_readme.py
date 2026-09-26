"""
The README's Python examples must run and print what they claim.

Each ```python block is executed in a fresh namespace. Full-line ``# ...``
comments directly after a ``print(...)`` line are the expected output.
"""

import contextlib
import io
import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


def _expected_output(block: str) -> list[str]:
    lines = block.splitlines()
    expected = []
    for i, line in enumerate(lines):
        if "print(" in line:
            j = i + 1
            while j < len(lines) and lines[j].startswith("# "):
                expected.append(lines[j][2:])
                j += 1
    return expected


@pytest.mark.parametrize("block", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_example(block, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(block, str(README), "exec"), {})
    expected = _expected_output(block)
    if expected:
        got = [line.strip() for line in out.getvalue().splitlines()]
        assert got[-len(expected):] == expected
