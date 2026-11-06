"""Sanity check for repository environment and packaging."""

import sys
from pathlib import Path


def test_python_version() -> None:
    """Ensure python version is 3.10+."""
    assert sys.version_info >= (3, 10), "Python 3.10 or higher is required."


def test_core_directories_exist() -> None:
    """Ensure essential project directories exist."""
    repo_root = Path(__file__).parent.parent
    expected_dirs = ["src", "configs", "data", "docs", "scripts", "tests", ".agents"]
    for dir_name in expected_dirs:
        assert (repo_root / dir_name).is_dir(), f"Expected directory {dir_name} missing"


def test_imports_succeed() -> None:
    """Ensure core package imports succeed without error."""
    import src
    import src.data
    import src.metrics
    import src.models
    import src.training
    import src.utils

    assert src is not None
    assert src.data is not None
    assert src.metrics is not None
    assert src.models is not None
    assert src.training is not None
    assert src.utils is not None
