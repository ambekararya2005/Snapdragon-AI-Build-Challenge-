# Kavach — project rules

Kavach is an offline, on-device scam shield for Windows PCs (Snapdragon AI Lab challenge).
It watches for remote-access tools, reads the active window with OCR, transcribes call audio,
and fuses these signals into a risk score that triggers a full-screen warning.

- Target: Snapdragon X HP PCs, ONNX Runtime `QNNExecutionProvider` (Hexagon NPU).
- Dev machine: Windows x64, Ryzen 7000 + RTX 4050, PowerShell, `onnxruntime-directml`.

## Environment
- Python 3.11, Windows-first. Give **PowerShell** commands, not bash.
- Dev setup: `powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1`, then `.\.venv\Scripts\Activate.ps1`.
- Tests: `python -m pytest -q`.

## Windows environment
- Smart App Control is on and blocks some unsigned native Python extensions
  ("An Application Control policy has blocked this file").
- When a package is blocked: try older versions or a pure-Python alternative, pin what works in
  `requirements.txt` with a comment, and tell the user.
- Never disable security settings (Smart App Control, Defender, WDAC, etc.) yourself.
- Known: `onnx` is pinned to 1.18.0 for this reason (1.20.1 and 1.23.0 are blocked).

## Privacy is the product
- Never write screenshots, OCR text, transcripts or audio to disk.
- Never add network calls to runtime code.
- Logs contain only derived info (labels, counts, scores, latencies) unless the config
  setting `privacy.debug_show_text` is `true`.
- The same rule applies to everything printed (CLI self-tests, dashboard, overlays): window titles,
  OCR text, transcripts and command lines go through `kavach_privacy` (`redact_title`,
  `redact_text`, `redact_cmdline`). By default show the process name + `<title hidden>`, and
  cmdlines as exe + flag names only.
- `privacy.write_raw_to_disk` must stay `false`; the config loader refuses to start if it is `true`.

## ONNX Runtime
- Every ONNX model is loaded through `models/runtime.py`. No other module creates
  `onnxruntime.InferenceSession` directly. The only exception is the third-party OCR backend
  (see `models/ocr.py`).
- The execution provider comes only from config (`runtime.provider: qnn | cuda | dml | cpu`),
  with fallback to CPU and a logged warning. Never assume a provider.
- Only one onnxruntime package may be installed at a time. `onnxruntime`, `onnxruntime-directml`,
  `onnxruntime-gpu` and `onnxruntime-qnn` conflict with each other. Never list onnxruntime in
  `requirements.txt`; install it per machine:
  - dev: `onnxruntime-directml`
  - Snapdragon: plain `onnxruntime` + the `onnxruntime-qnn` 2.x plugin EP (not a separate build, so the pair does
    not conflict) on native ARM64 Python: `requirements-snapdragon.txt`, `config.snapdragon.yaml`
  - fallback: `onnxruntime`

## Layout
- Model weights go in `weights/` (gitignored). `models/` holds code only.
- Pipeline: capture → models → detect → fusion → act. Keep modules small and independent.
- Every module has a `python -m <module>` self-test entry point, plus light pytest tests in
  `tests/` that run without real models or Windows-only hardware where possible.
- Config lives in `config.yaml`, loaded via `kavach_config.get_config()`.
  Env overrides: `KAVACH_PROVIDER`, `KAVACH_CONFIG` (alternate config path).
