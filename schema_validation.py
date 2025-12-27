import json
from jsonschema import Draft202012Validator


def _load_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_dashboard_data(data: dict, schema_path: str = "schemas/dashboard_data.schema.json") -> None:
    Draft202012Validator(_load_schema(schema_path)).validate(data)


def validate_governance_report(data: dict, schema_path: str = "schemas/governance_report.schema.json") -> None:
    Draft202012Validator(_load_schema(schema_path)).validate(data)


def validate_policy_snapshot(data: dict, schema_path: str = "schemas/policy_snapshot.schema.json") -> None:
    Draft202012Validator(_load_schema(schema_path)).validate(data)
