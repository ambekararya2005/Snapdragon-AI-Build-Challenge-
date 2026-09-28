# Kavach

This is an offline, on-device scam shield for Windows PCs, built for the Snapdragon AI Lab challenge.

It watches for remote-access tools, reads the active window with OCR, transcribes call audio, and fuses these signals into a risk score. A high score triggers a full-screen warning. Nothing leaves the device.

## Dev setup (Windows x64)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1
.\.venv\Scripts\Activate.ps1
python -m kavach_config
python -m pytest -q
```

See `CLAUDE.md` for project rules. TODO: architecture, Snapdragon setup, demo.
