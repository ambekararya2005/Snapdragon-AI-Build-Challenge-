import importlib

import pytest

MODULES = [
    "main",
    "capture.screen", "capture.audio", "capture.processes",
    "models.runtime", "models.ocr", "models.asr",
    "detect.screen_classifier", "detect.intent",
    "fusion.risk",
    "act.overlay", "act.remote_control",
    "dashboard.app",
    "bench.aihub_profile",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name).__doc__


def test_main_refuses_strict_npu_config_without_the_provider(monkeypatch, capsys):
    """python main.py --config config.snapdragon.yaml on a PC without QNN: loud error, exit 2, nothing started."""
    import main
    from kavach_config import get_config
    from models import runtime

    monkeypatch.setattr(runtime, "available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr("kavach.pipeline.Pipeline.start", lambda self: (_ for _ in ()).throw(AssertionError("started")))
    try:
        assert main.main(["--config", "config.snapdragon.yaml", "--plain", "--seconds", "1"]) == 2
    finally:
        monkeypatch.delenv("KAVACH_CONFIG", raising=False)
        get_config.cache_clear()
    err = capsys.readouterr().err
    assert "Kavach cannot start" in err and "QNNExecutionProvider" in err and "Refusing to start" in err
