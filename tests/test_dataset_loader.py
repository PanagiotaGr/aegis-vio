from pathlib import Path

from src.dataset_loader import EuRoCDatasetLoader


def test_dataset_loader_raises_for_missing_path(tmp_path: Path):
    missing = tmp_path / "missing_sequence"
    try:
        EuRoCDatasetLoader(missing)
    except FileNotFoundError:
        assert True
    else:
        assert False
