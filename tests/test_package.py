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
