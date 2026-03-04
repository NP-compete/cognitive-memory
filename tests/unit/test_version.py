"""Test package version and imports."""

import cognitive_memory


def test_version_exists() -> None:
    """Verify version string is defined."""
    assert hasattr(cognitive_memory, "__version__")
    assert isinstance(cognitive_memory.__version__, str)


def test_version_format() -> None:
    """Verify version follows semver format."""
    version = cognitive_memory.__version__
    parts = version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
