import importlib
import logging
from types import SimpleNamespace

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


def test_external_stats_in_registry():
    lat = runtime.LatencyStats()
    runtime.register_external_stats("ext", lambda: {"requested_provider": "x", "actual_provider": "y", **lat.summary()})
    assert runtime.all_stats() == [{"name": "ext", "requested_provider": "x", "actual_provider": "y",
                                    "n": 0, "last_ms": None, "p50_ms": None, "p95_ms": None}]
    for ms in (10, 20, 30):
        lat.record(ms)
    (s,) = runtime.all_stats()
    assert s["n"] == 3 and s["last_ms"] == 30 and s["p50_ms"] == 20

    runtime.register_external_stats("broken", lambda: 1 / 0)       # a failing provider is skipped
    assert [s["name"] for s in runtime.all_stats()] == ["ext"]
    runtime.unregister_external_stats("ext")
    runtime.unregister_external_stats("broken")
    assert runtime.all_stats() == []


def test_dml_sessions_share_one_run_lock():
    class Fake:
        def __init__(self, prov):
            self.prov = prov

        def get_providers(self):
            return [self.prov, "CPUExecutionProvider"]

        def run(self, names, feeds):
            return [feeds["x"]]

    a = runtime.KavachSession(Fake("DmlExecutionProvider"), "a", "DmlExecutionProvider")
    b = runtime.KavachSession(Fake("DmlExecutionProvider"), "b", "DmlExecutionProvider")
    c = runtime.KavachSession(Fake("CPUExecutionProvider"), "c", "CPUExecutionProvider")
    assert a._run_lock is b._run_lock is runtime._DML_RUN_LOCK     # concurrent DML runs segfault: serialize
    assert c._run_lock is not runtime._DML_RUN_LOCK
    with runtime._DML_RUN_LOCK:                                      # a DML run waits for the lock ...
        import threading
        done = []
        t = threading.Thread(target=lambda: done.append(a.run({"x": 1})))
        t.start()
        t.join(0.2)
        assert not done
        assert c.run({"x": 2}) == [2]                                # ... a CPU run does not
    t.join(2)
    assert done == [[1]]


def test_ort_session_proxy_delegates(dummy_model):
    ks = runtime.create_session(dummy_model, "proxy", "cpu", RT_CFG)
    proxy = ks.as_ort_session()                        # what rapidocr gets instead of an InferenceSession
    feeds = ks.dummy_feeds()
    name = proxy.get_inputs()[0].name
    out = proxy.run(None, {name: feeds[name]})
    assert np.array_equal(out[0], ks.session.run(None, feeds)[0])
    assert proxy.get_providers()[0] == "CPUExecutionProvider"
    assert ks.stats()["n"] == 1                        # proxied runs are counted by the wrapper


# ---------------------------------------------------------------- strict mode + QNN plugin EP (faked onnxruntime)

QNN_STRICT = {"provider": "qnn", "fallback_to_cpu": False, "qnn": {"backend_path": "QnnHtp.dll"}}


@pytest.fixture
def no_plugins(monkeypatch):
    monkeypatch.setattr(runtime, "_plugins_tried", True)
    monkeypatch.setattr(runtime, "_plugins_registered", {})


def test_strict_qnn_without_provider_is_a_clear_startup_error(no_plugins, monkeypatch, dummy_model):
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    msg = runtime.provider_problem(QNN_STRICT)
    assert "QNNExecutionProvider" in msg and "fallback_to_cpu: false" in msg and "onnxruntime-qnn" in msg
    assert runtime.provider_problem({**QNN_STRICT, "fallback_to_cpu": True}) is None     # lenient: falls back
    with pytest.raises(runtime.ProviderUnavailable):
        runtime.create_session(dummy_model, "strict", runtime_cfg=QNN_STRICT)


class _FakePlugin:
    @staticmethod
    def get_ep_name():
        return "QNNExecutionProvider"

    @staticmethod
    def get_library_path():
        return r"C:\fake\onnxruntime_qnn\onnxruntime_providers_qnn.dll"

    @staticmethod
    def get_qnn_htp_path():
        return r"C:\fake\onnxruntime_qnn\QnnHtp.dll"


def test_qnn_plugin_ep_is_registered_and_attached_per_device(monkeypatch, dummy_model):
    """onnxruntime-qnn 2.x: registered from its library, attached with add_provider_for_devices, strict."""
    monkeypatch.setattr(runtime, "_plugins_tried", False)
    monkeypatch.setattr(runtime, "_plugins_registered", {})
    registered, attached, created = [], [], []
    npu = SimpleNamespace(type="NPU")
    devices = [SimpleNamespace(ep_name="CPUExecutionProvider", device=SimpleNamespace(type="CPU")),
               SimpleNamespace(ep_name="QNNExecutionProvider", device=npu)]
    monkeypatch.setattr(runtime.importlib, "import_module",
                        lambda name: _FakePlugin if name == "onnxruntime_qnn" else importlib.import_module(name))
    monkeypatch.setattr(runtime.ort, "register_execution_provider_library", lambda n, p: registered.append((n, p)),
                        raising=False)
    monkeypatch.setattr(runtime.ort, "get_ep_devices", lambda: devices, raising=False)
    monkeypatch.setattr(runtime.ort, "OrtHardwareDeviceType", SimpleNamespace(NPU="NPU"), raising=False)
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

    class FakeOptions:
        def __init__(self):
            self.entries = {}
            self.graph_optimization_level = None

        def add_session_config_entry(self, k, v):
            self.entries[k] = v

        def add_provider_for_devices(self, devs, opts):
            attached.append((devs, opts))

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            created.append((path, sess_options, providers))

        def get_providers(self):
            return ["QNNExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(runtime.ort, "SessionOptions", FakeOptions)
    monkeypatch.setattr(runtime.ort, "InferenceSession", FakeSession)
    assert runtime.available_providers() == ["CPUExecutionProvider", "QNNExecutionProvider"]
    assert runtime.provider_problem(QNN_STRICT) is None
    s = runtime.create_session(dummy_model, "npu", runtime_cfg=QNN_STRICT)
    assert registered == [("QNNExecutionProvider", _FakePlugin.get_library_path())]
    (devs, opts), = attached
    assert [d.ep_name for d in devs] == ["QNNExecutionProvider"]
    assert opts["backend_path"] == _FakePlugin.get_qnn_htp_path()             # resolved inside the plugin
    _, so, providers = created[0]
    assert providers is None and so.entries == {"session.disable_cpu_ep_fallback": "1"}
    assert s.actual_provider == "QNNExecutionProvider"


def test_input_dtypes_for_fp16_compiled_models(dummy_model):
    sess = runtime.create_session(dummy_model, "dummy", provider="cpu", runtime_cfg=RT_CFG)
    assert sess.input_dtypes() == {"input": np.float32}
