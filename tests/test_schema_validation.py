
import json
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path("schemas/dashboard_data.schema.json")
DATA_PATH = Path("output/dashboard_data.json")

def test_dashboard_data_matches_schema():
    assert SCHEMA_PATH.exists(), "Schema file missing"
    assert DATA_PATH.exists(), "Dashboard data file missing"

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    with open(DATA_PATH) as f:
        data = json.load(f)

    # Validate
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        pytest.fail(f"Schema Validation Failed: {e}")
