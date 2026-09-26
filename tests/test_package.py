"""Package-level metadata tests."""

import importlib.metadata

import vd


def test_version_matches_installed_distribution():
    """``vd.__version__`` must track the released version, not a stale literal."""
    assert vd.__version__ == importlib.metadata.version("vd")


def test_bundled_skills_are_spec_clean():
    """Each bundled skill's frontmatter name matches its folder; audience in metadata."""
    import re

    skills = sorted(vd.skills_dir().glob("*/SKILL.md"))
    assert len(skills) >= 7
    for path in skills:
        front = path.read_text().split("---")[1]
        assert re.search(rf"^name: {path.parent.name}$", front, re.M), path
        assert not re.search(r"^audience:", front, re.M), path


def test_sdist_ships_the_bundled_skills():
    """The sdist (which the wheel is built from) must carry vd/data/skills (#28).

    ``.claude/skills/*`` symlink into ``vd/data/skills/``; if hatch walks
    ``.claude`` it keeps those copies and silently drops the real ones.
    """
    import pathlib

    import pytest

    sdist = pytest.importorskip("hatchling.builders.sdist")
    root = pathlib.Path(__file__).resolve().parent.parent
    paths = {  # hatchling joins with os.sep; compare POSIX-style (Windows CI)
        f.distribution_path.replace("\\", "/")
        for f in sdist.SdistBuilder(str(root)).recurse_included_files()
    }
    expected = {
        f"vd/data/skills/{p.parent.name}/SKILL.md"
        for p in vd.skills_dir().glob("*/SKILL.md")
    }
    assert expected <= paths
    assert "vd/data/providers.yaml" in paths
    assert not any(p.startswith(".claude") for p in paths)
