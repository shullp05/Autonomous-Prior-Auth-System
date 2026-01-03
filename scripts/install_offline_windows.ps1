param(
    [string]$RequirementsFile = "requirements.txt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$WheelDir = "artifacts\wheels\windows"

if (-not (Test-Path -Path $WheelDir)) {
    throw "Wheelhouse not found: $WheelDir"
}

$env:PIP_NO_INDEX = "1"
$env:PIP_FIND_LINKS = $WheelDir

python -m pip install --no-index --find-links $WheelDir -r $RequirementsFile
