
import json
import os
from pathlib import Path

import jsonschema
import pytest

from priorauth import paths

SCHEMA_PATH = paths.SCHEMAS_DIR / "dashboard_data.schema.json"
DATA_PATH = Path(os.getenv("PA_DASHBOARD_DATA_PATH", str(paths.OUTPUT_DIR / "dashboard_data.json")))
FIXTURE_PATH = paths.REPO_ROOT / "tests" / "fixtures" / "dashboard_data.json"

def test_dashboard_data_matches_schema():
    assert SCHEMA_PATH.exists(), "Schema file missing"
    data_path = DATA_PATH if DATA_PATH.exists() else FIXTURE_PATH
    assert data_path.exists(), "Dashboard data file missing"

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    with open(data_path) as f:
        data = json.load(f)

    # Validate
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        pytest.fail(f"Schema Validation Failed: {e}")
