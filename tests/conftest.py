import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
# isolated DB / folders for API tests (must be set before the app is imported)
_tmp = tempfile.mkdtemp(prefix="flfraud_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ.setdefault("RESULTS_DIR", f"{_tmp}/results")
os.environ.setdefault("DATA_DIR", f"{_tmp}/data")

import pytest  # noqa: E402

from src.data.loader import generate_synthetic  # noqa: E402
from src.data.preprocessor import prepare  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    return generate_synthetic(6000, 0.02, seed=1)


@pytest.fixture(scope="session")
def prepared(small_df):
    return prepare(small_df, seed=1)
