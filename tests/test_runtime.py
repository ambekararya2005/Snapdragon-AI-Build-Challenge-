import logging

import numpy as np
import onnxruntime as ort
import pytest

from models import runtime
from scripts.make_dummy_model import build_dummy_model

RT_CFG = {"provider": "cpu", "fallback_to_cpu": True, "log_latency": False, "qnn": {"backend_path": "QnnHtp.dll"}}


@pytest.fixture
def dummy_model(tmp_path):
    return build_dummy_model(tmp_path / "dummy.onnx")


@pytest.fixture(autouse=True)
def clean_registry():
    runtime.clear_registry()
    yield
    runtime.clear_registry()


def test_provider_mapping():
    cfg = {"qnn": {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"},
           "dml": {"device_id": 0}}
    name, opts = runtime.provider_config("qnn", cfg)
    assert name == "QNNExecutionProvider"
    assert opts == {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"}
    assert runtime.provider_config("dml", cfg) == ("DmlExecutionProvider", {"device_id": "0"})
    assert runtime.provider_config("cpu", cfg) == ("CPUExecutionProvider", {})
    assert runtime._providers_list("DmlExecutionProvider", {})[-1][0] == "CPUExecutionProvider"
    with pytest.raises(ValueError):
        runtime.provider_config("tpu", cfg)


def test_dummy_model_runs_on_cpu(dummy_model):
    sess = runtime.create_session(dummy_model, "dummy", provider="cpu", runtime_cfg=RT_CFG)
    assert sess.actual_provider == "CPUExecutionProvider"
    assert sess.input_specs() == [{"name": "input", "shape": [1, 3, 224, 224], "dtype": "tensor(float)"}]
    (logits,) = sess.run({"input": np.random.rand(1, 3, 224, 224).astype(np.float32)})
    assert logits.shape == (1, 10)


def test_qnn_falls_back_to_cpu_with_warning(dummy_model, caplog):
    if "QNNExecutionProvider" in ort.get_available_providers():
        pytest.skip("QNN is available on this machine")
    with caplog.at_level(logging.WARNING, logger="models.runtime"):
        sess = runtime.create_session(dummy_model, "dummy_qnn", provider="qnn", runtime_cfg=RT_CFG)
    assert sess.requested_provider == "QNNExecutionProvider"
    assert sess.actual_provider == "CPUExecutionProvider"
    assert any("QNNExecutionProvider" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_qnn_without_fallback_raises(dummy_model):
    if "QNNExecutionProvider" in ort.get_available_providers():
        pytest.skip("QNN is available on this machine")
    with pytest.raises(RuntimeError, match="not available"):
        runtime.create_session(dummy_model, "x", provider="qnn", runtime_cfg={**RT_CFG, "fallback_to_cpu": False})


def test_stats_and_registry(dummy_model):
    sess = runtime.create_session(dummy_model, "dummy", provider="cpu", runtime_cfg=RT_CFG)
    assert sess.stats()["n"] == 0 and sess.stats()["p50_ms"] is None
    sess.warmup(n=5)
    s = sess.stats()
    assert s["name"] == "dummy"
    assert s["requested_provider"] == s["actual_provider"] == "CPUExecutionProvider"
    assert s["n"] == 5
    assert s["last_ms"] > 0 and 0 < s["p50_ms"] <= s["p95_ms"]
    assert runtime.all_stats() == [s]
    assert runtime.get_session("dummy") is sess


def test_warmup_overrides_dynamic_dims():
    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            class I:
                name, shape, type = "x", ["batch", 3, None], "tensor(float)"
            return [I()]

    ks = runtime.KavachSession(FakeSession(), "fake", "CPUExecutionProvider")
    assert ks.dummy_feeds()["x"].shape == (1, 3, 1)
    assert ks.dummy_feeds({"x": [2, 3, 7]})["x"].shape == (2, 3, 7)


def test_system_info():
    info = runtime.system_info()
    assert info["onnxruntime_version"] == ort.__version__
    assert "CPUExecutionProvider" in info["available_providers"]
    assert info["onnxruntime_packages"]
    assert set(info) >= {"machine", "python"}
