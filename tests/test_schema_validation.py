
import json
import os
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path("schemas/dashboard_data.schema.json")
DATA_PATH = Path(os.getenv("PA_DASHBOARD_DATA_PATH", "output/dashboard_data.json"))
FIXTURE_PATH = Path("tests/fixtures/dashboard_data.json")

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
