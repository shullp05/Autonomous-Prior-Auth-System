"""Project path helpers anchored to the repo root.

Use these to avoid cwd-sensitive paths across scripts, tests, and Docker runs.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
REPO_ROOT = SRC_DIR.parent

DATA_DIR = Path(os.getenv("PA_DATA_DIR", REPO_ROOT)).resolve()
OUTPUT_DIR = Path(os.getenv("PA_OUTPUT_DIR", REPO_ROOT / "output")).resolve()
CHROMA_DIR = Path(os.getenv("PA_CHROMA_DIR", REPO_ROOT / "chroma_db")).resolve()
POLICIES_DIR = Path(os.getenv("PA_POLICIES_DIR", REPO_ROOT / "policies")).resolve()
SCHEMAS_DIR = Path(os.getenv("PA_SCHEMAS_DIR", REPO_ROOT / "schemas")).resolve()
MODELS_DIR = Path(os.getenv("PA_MODELS_DIR", REPO_ROOT / "models")).resolve()
UI_DIR = Path(os.getenv("PA_UI_DIR", REPO_ROOT / "apps" / "ui")).resolve()
DOCS_DIR = Path(os.getenv("PA_DOCS_DIR", REPO_ROOT / "docs")).resolve()

POLICY_PDF = Path(os.getenv("PA_POLICY_PDF", REPO_ROOT / "Policy_Weight_Mgmt_2025.pdf")).resolve()
GUIDELINE_PATH = Path(os.getenv("PA_GUIDELINE_PATH", REPO_ROOT / "UpdatedPAGuidelines.txt")).resolve()
POLICY_SNAPSHOT_PATH = Path(
    os.getenv("PA_SNAPSHOT_PATH", POLICIES_DIR / "RX-WEG-2025.json")
).resolve()
