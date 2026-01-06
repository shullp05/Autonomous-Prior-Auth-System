import json

from jsonschema import Draft202012Validator

from priorauth import paths


def _load_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_schema_path(filename: str) -> str:
    return str(paths.SCHEMAS_DIR / filename)


def validate_dashboard_data(
    data: dict,
    schema_path: str = _default_schema_path("dashboard_data.schema.json"),
) -> None:
    Draft202012Validator(_load_schema(schema_path)).validate(data)


def validate_governance_report(
    data: dict,
    schema_path: str = _default_schema_path("governance_report.schema.json"),
) -> None:
    Draft202012Validator(_load_schema(schema_path)).validate(data)


def validate_policy_snapshot(
    data: dict,
    schema_path: str = _default_schema_path("policy_snapshot.schema.json"),
) -> None:
    Draft202012Validator(_load_schema(schema_path)).validate(data)
