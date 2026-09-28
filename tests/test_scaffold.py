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
