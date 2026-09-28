# Kavach dev setup (Windows x64 dev machine).
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1
#
# rapidocr_onnxruntime pulls in plain `onnxruntime`, which conflicts with onnxruntime-directml.
# So we install requirements first, remove every onnxruntime* package, then install
# onnxruntime-directml last.

param(
    [string]$OrtPackage = "onnxruntime-directml"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv with Python 3.11..."
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "py -3.11 -m venv failed (is Python 3.11 installed?)" }
}
$py = Join-Path $root ".venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "Installing requirements.txt..."
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }

Write-Host "Removing all onnxruntime* packages..."
$ortPkgs = & $py -m pip list --format=freeze |
    ForEach-Object { ($_ -split "==")[0] } |
    Where-Object { $_ -like "onnxruntime*" -and $_ -ne "rapidocr_onnxruntime" -and $_ -ne "rapidocr-onnxruntime" }
foreach ($p in $ortPkgs) {
    Write-Host "  uninstalling $p"
    & $py -m pip uninstall -y $p
    if ($LASTEXITCODE -ne 0) { throw "failed to uninstall $p" }
}

Write-Host "Installing $OrtPackage (last, so nothing overrides it)..."
& $py -m pip install $OrtPackage
if ($LASTEXITCODE -ne 0) { throw "pip install $OrtPackage failed" }

Write-Host "Installed onnxruntime packages:"
& $py -m pip list --format=freeze | Where-Object { $_ -like "onnxruntime*" }

Write-Host "Available providers:"
& $py -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
