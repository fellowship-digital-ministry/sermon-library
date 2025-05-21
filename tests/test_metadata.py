import pytest

from api.metadata_utils import (
    get_all_sermon_metadata,
    get_sermon_metadata,
    get_metadata_directory,
)


def test_metadata_directory_found():
    """Metadata directory should exist and contain files."""
    metadata_dir = get_metadata_directory()
    assert metadata_dir is not None


def test_load_all_metadata():
    """All sermon metadata should load without errors."""
    data = get_all_sermon_metadata()
    assert isinstance(data, dict)
    assert len(data) > 0

    sample_id = next(iter(data))
    assert get_sermon_metadata(sample_id) == data[sample_id]
