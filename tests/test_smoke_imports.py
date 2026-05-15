import aml_inspector
from aml_inspector.config import DATA_PROCESSED, FEAST_REPO, PROJECT_ROOT


def test_import_package():
    assert aml_inspector.__version__


def test_paths_exist():
    assert PROJECT_ROOT.is_dir()
    assert FEAST_REPO.is_dir()
    assert (FEAST_REPO / "feature_store.yaml").is_file()
    assert DATA_PROCESSED.is_dir()
