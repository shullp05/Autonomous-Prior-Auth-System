import json

from jsonschema import Draft202012Validator


def validate_dashboard_data(data: dict, schema_path: str = "schemas/dashboard_data.schema.json") -> None:
    with open(schema_path) as f:
        schema = json.load(f)
    Draft202012Validator(schema).validate(data)

def validate_governance_report(data: dict, schema_path: str = "schemas/governance_report.schema.json") -> None:
    with open(schema_path) as f:
        schema = json.load(f)
    Draft202012Validator(schema).validate(data)


def validate_policy_snapshot(data: dict, schema_path: str = "schemas/policy_snapshot.schema.json") -> None:
    with open(schema_path) as f:
        schema = json.load(f)
    Draft202012Validator(schema).validate(data)
