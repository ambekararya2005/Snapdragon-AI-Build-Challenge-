"""Single entry point for creating ONNX Runtime sessions with the configured execution provider (qnn | cuda | dml | cpu), falling back to CPU with a logged warning.

    from models import runtime
    sess = runtime.create_session("weights/foo.onnx", "foo")   # provider from config.runtime.provider
    outputs = sess.run({"input": x})
    runtime.all_stats()                                         # latency stats for every session

Self-test / CLI:
    python -m models.runtime --info
    python -m models.runtime --model weights/dummy.onnx --provider dml --runs 50
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import onnxruntime as ort

log = logging.getLogger("models.runtime")

PROVIDER_NAMES = {
    "qnn": "QNNExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
CPU = PROVIDER_NAMES["cpu"]
ORT_PACKAGES = ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-qnn")
# onnxruntime-qnn >= 2.0 is a plugin EP meant to sit next to plain onnxruntime (not a conflicting build).
STATS_WINDOW = 100

_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(bool)": np.bool_,
}


def _runtime_cfg() -> Mapping[str, Any]:
    from kavach_config import get_config

    return get_config().runtime


def provider_config(key: str, runtime_cfg: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Map a config provider key (qnn | dml | cuda | cpu) to (ORT provider name, provider options)."""
    key = key.strip().lower()
    if key not in PROVIDER_NAMES:
        raise ValueError(f"unknown provider {key!r}; expected one of {tuple(PROVIDER_NAMES)}")
    if key == "qnn":
        q = runtime_cfg.get("qnn", {})
        opts = {
            "backend_path": q.get("backend_path", "QnnHtp.dll"),
            "htp_performance_mode": q.get("htp_performance_mode", "burst"),
            "enable_htp_fp16_precision": q.get("enable_htp_fp16_precision", "1"),
        }
    elif key in ("dml", "cuda"):
        opts = {"device_id": runtime_cfg.get(key, {}).get("device_id", 0)}
    else:
        opts = {}
    return PROVIDER_NAMES[key], {k: str(v) for k, v in opts.items()}


class ProviderUnavailable(RuntimeError):
    """The configured provider is missing and runtime.fallback_to_cpu is false."""


# ---------------------------------------------------------------- plugin execution providers
# onnxruntime-qnn 2.x is a *plugin* EP (bundles QAIRT 2.50, matching the AI Hub compile) installed next to
# plain onnxruntime >= 1.24.2: it is registered from its library and attached to sessions per device
# (SessionOptions.add_provider_for_devices); it does not appear in ort.get_available_providers().
# onnxruntime-qnn 1.x (a full onnxruntime build, QAIRT 2.42) and DirectML / CUDA / CPU use the classic path.
# PREPARED, NOT YET RUN ON A SNAPDRAGON DEVICE: covered by unit tests with a faked onnxruntime only.

_PLUGIN_MODULES = {PROVIDER_NAMES["qnn"]: "onnxruntime_qnn"}
_plugin_lock = threading.Lock()
_plugins_registered: dict[str, Any] = {}               # provider name -> plugin module (registered once)
_plugins_tried = False


def register_plugin_eps() -> list[str]:
    """Register the installed plugin EPs with onnxruntime once. Returns the registered provider names;
    empty when no plugin package is installed or onnxruntime predates plugin EPs."""
    global _plugins_tried
    with _plugin_lock:
        if not _plugins_tried:
            _plugins_tried = True
            if hasattr(ort, "register_execution_provider_library"):
                for provider, module in _PLUGIN_MODULES.items():
                    try:
                        mod = importlib.import_module(module)
                    except ImportError:
                        continue
                    try:
                        ort.register_execution_provider_library(mod.get_ep_name(), mod.get_library_path())
                        _plugins_registered[provider] = mod
                        log.info("registered plugin EP %s from %s", provider, module)
                    except Exception as e:  # noqa: BLE001
                        log.warning("plugin EP %s (%s) could not be registered: %s: %s", provider, module,
                                    type(e).__name__, e)
        return list(_plugins_registered)


def plugin_ep_devices(provider_name: str) -> list[Any]:
    register_plugin_eps()
    if not hasattr(ort, "get_ep_devices"):
        return []
    return [d for d in ort.get_ep_devices() if getattr(d, "ep_name", None) == provider_name]


def _is_npu(ep_device: Any) -> bool:
    npu = getattr(getattr(ort, "OrtHardwareDeviceType", None), "NPU", None)
    return npu is not None and getattr(getattr(ep_device, "device", None), "type", None) == npu


def available_providers() -> list[str]:
    """Built-in providers of the installed onnxruntime plus registered plugin EPs that have a device."""
    names = list(ort.get_available_providers())
    for provider in register_plugin_eps():
        if provider not in names and plugin_ep_devices(provider):
            names.append(provider)
    return names


def _plugin_options(provider_name: str, options: Mapping[str, str]) -> dict[str, str]:
    """QNN plugin: a bare backend_path (QnnHtp.dll) is resolved inside the plugin package, which ships it."""
    opts = dict(options)
    mod = _plugins_registered.get(provider_name)
    bp = opts.get("backend_path")
    if mod is not None and bp and Path(bp).name == bp and hasattr(mod, "get_qnn_htp_path") \
            and bp.lower() == "qnnhtp.dll":
        opts["backend_path"] = mod.get_qnn_htp_path()
    return opts


def installed_ort_packages() -> list[str]:
    out = []
    for pkg in ORT_PACKAGES:
        try:
            out.append(f"{pkg}=={importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def provider_problem(runtime_cfg: Mapping[str, Any]) -> str | None:
    """Why a strict config (fallback_to_cpu: false) cannot run here, or None. Used by main.py before
    anything starts, so a missing NPU provider is a loud startup error, never a quiet CPU run."""
    key = str(runtime_cfg.get("provider", "cpu")).strip().lower()
    requested, _ = provider_config(key, runtime_cfg)
    available = available_providers()
    if requested in available or bool(runtime_cfg.get("fallback_to_cpu", True)):
        return None
    pkgs = ", ".join(installed_ort_packages()) or "none"
    hint = ("install onnxruntime-qnn on native ARM64 Python 3.11 on a Snapdragon X PC (requirements-snapdragon.txt)"
            if key == "qnn" else f"install the onnxruntime package that provides {requested}")
    return (f"runtime.provider is {key!r} ({requested}) with fallback_to_cpu: false, but this onnxruntime "
            f"({pkgs}, {platform.machine()}) only has {available}. Refusing to start rather than run on the CPU. "
            f"Fix: {hint}, or use a config with a provider available here (config.yaml).")


def _session_options(provider_name: str, strict: bool = False) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if strict and provider_name != CPU:
        # fallback_to_cpu: false also forbids ORT from placing single unsupported operators on the CPU.
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if provider_name == PROVIDER_NAMES["dml"]:
        # DirectML does not support memory pattern optimisation or parallel execution.
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def _providers_list(provider_name: str, options: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    if provider_name == CPU:
        return [(CPU, {})]
    return [(provider_name, options), (CPU, {})]


class LatencyStats:
    """Thread-safe rolling latency window (last STATS_WINDOW samples) plus a total count."""

    def __init__(self, window: int = STATS_WINDOW):
        self._lat_ms: deque[float] = deque(maxlen=window)
        self._n = 0
        self._lock = threading.Lock()

    def record(self, ms: float) -> None:
        with self._lock:
            self._lat_ms.append(float(ms))
            self._n += 1

    def reset(self) -> None:
        with self._lock:
            self._lat_ms.clear()
            self._n = 0

    def summary(self) -> dict[str, Any]:
        """{n, last_ms, p50_ms, p95_ms}; None values when there are no samples yet."""
        with self._lock:
            lat = np.array(self._lat_ms, dtype=np.float64)
            n = self._n
        return {
            "n": n,
            "last_ms": round(float(lat[-1]), 3) if lat.size else None,
            "p50_ms": round(float(np.percentile(lat, 50)), 3) if lat.size else None,
            "p95_ms": round(float(np.percentile(lat, 95)), 3) if lat.size else None,
        }


# DirectML: Run() on two DML sessions from two threads at the same time segfaults
# (onnxruntime-directml 1.24.4, dev PC; OCR and ASR threads in scripts/signals_console.py).
# All DML runs in the process are serialized; latency is measured after the lock is acquired.
_DML_RUN_LOCK = threading.Lock()


class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


class KavachSession:
    """Wraps an onnxruntime.InferenceSession and records per-run latency."""

    def __init__(self, session: ort.InferenceSession, name: str, requested_provider: str, log_latency: bool = False):
        self.session = session
        self.name = name
        self.requested_provider = requested_provider
        self.actual_provider = session.get_providers()[0]
        self.log_latency = log_latency
        self.latency = LatencyStats()
        self._run_lock = _DML_RUN_LOCK if self.actual_provider == PROVIDER_NAMES["dml"] else _NoLock()

    def run(self, feeds: Mapping[str, np.ndarray], output_names: Sequence[str] | None = None) -> list[np.ndarray]:
        with self._run_lock:
            t0 = time.perf_counter()
            outputs = self.session.run(list(output_names) if output_names else None, dict(feeds))
            ms = (time.perf_counter() - t0) * 1000.0
        self.latency.record(ms)
        if self.log_latency:
            log.debug("%s: run %.2f ms on %s", self.name, ms, self.actual_provider)
        return outputs

    def as_ort_session(self) -> "OrtSessionProxy":
        """InferenceSession look-alike for third-party code (rapidocr): run() goes through this wrapper,
        so the DML lock and latency stats apply."""
        return OrtSessionProxy(self)

    def input_specs(self) -> list[dict[str, Any]]:
        return [{"name": i.name, "shape": list(i.shape), "dtype": i.type} for i in self.session.get_inputs()]

    def input_dtypes(self) -> dict[str, Any]:
        """Input name -> numpy dtype. Compiled NPU models (AI Hub precompiled_qnn_onnx) often take float16
        where the plain export takes float32; callers cast their feeds with this."""
        return {i.name: _ORT_DTYPES[i.type] for i in self.session.get_inputs() if i.type in _ORT_DTYPES}

    def output_specs(self) -> list[dict[str, Any]]:
        return [{"name": o.name, "shape": list(o.shape), "dtype": o.type} for o in self.session.get_outputs()]

    def dummy_feeds(self, overrides: Mapping[str, Sequence[int]] | None = None) -> dict[str, np.ndarray]:
        """Zero-filled inputs from the input specs; dynamic dims become 1 unless a shape override is given."""
        overrides = overrides or {}
        feeds = {}
        for spec in self.input_specs():
            if spec["name"] in overrides:
                shape = tuple(overrides[spec["name"]])
            else:
                shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in spec["shape"])
            dtype = _ORT_DTYPES.get(spec["dtype"])
            if dtype is None:
                raise TypeError(f"{self.name}: no dummy input for {spec['name']} of type {spec['dtype']}")
            feeds[spec["name"]] = np.zeros(shape, dtype=dtype)
        return feeds

    def warmup(self, n: int = 3, overrides: Mapping[str, Sequence[int]] | None = None) -> None:
        feeds = self.dummy_feeds(overrides)
        for _ in range(n):
            self.run(feeds)

    def reset_stats(self) -> None:
        self.latency.reset()

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requested_provider": self.requested_provider,
            "actual_provider": self.actual_provider,
            **self.latency.summary(),
        }


class OrtSessionProxy:
    """Duck-typed onnxruntime.InferenceSession backed by a KavachSession."""

    def __init__(self, ks: KavachSession):
        self._ks = ks

    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray], run_options: Any = None):
        return self._ks.run(input_feed, output_names)

    def __getattr__(self, name: str) -> Any:              # get_inputs, get_outputs, get_providers, ...
        return getattr(self._ks.session, name)


_registry: dict[str, KavachSession] = {}
_external: dict[str, Callable[[], dict[str, Any]]] = {}
_registry_lock = threading.Lock()


def register_external_stats(name: str, stats_fn: Callable[[], dict[str, Any]]) -> None:
    """Add stats for inference not run through create_session (e.g. the rapidocr backend).

    stats_fn() should return the same fields as KavachSession.stats():
    {name, requested_provider, actual_provider, n, last_ms, p50_ms, p95_ms}.
    """
    with _registry_lock:
        _external[name] = stats_fn


def unregister_external_stats(name: str) -> None:
    with _registry_lock:
        _external.pop(name, None)


def create_session(
    model_path: str | Path,
    name: str,
    provider: str | None = None,
    runtime_cfg: Mapping[str, Any] | None = None,
) -> KavachSession:
    """Create a session on `provider` (default: config runtime.provider), falling back to CPU if allowed."""
    rt = _runtime_cfg() if runtime_cfg is None else runtime_cfg
    key = (provider or rt.get("provider", "cpu")).strip().lower()
    fallback = bool(rt.get("fallback_to_cpu", True))
    requested, options = provider_config(key, rt)
    model_path = str(model_path)

    available = available_providers()
    use = requested
    if requested not in available:
        if not fallback:
            raise ProviderUnavailable(f"{name}: requested {requested} is not available (available: {available}) "
                                      f"and runtime.fallback_to_cpu is false")
        log.warning("%s: requested %s is not available (available: %s); using %s", name, requested, available, CPU)
        use = CPU

    try:
        so = _session_options(use, strict=not fallback)
        if use != CPU and use not in ort.get_available_providers():
            # Plugin EP (onnxruntime-qnn 2.x on top of onnxruntime): attached per device, not by name.
            devices = plugin_ep_devices(use)
            npu = [d for d in devices if _is_npu(d)] or devices
            so.add_provider_for_devices(npu, _plugin_options(use, options))
            session = ort.InferenceSession(model_path, sess_options=so)
        else:
            session = ort.InferenceSession(model_path, sess_options=so, providers=_providers_list(use, options))
    except Exception as e:
        if use == CPU or not fallback:
            raise
        log.warning("%s: session creation on %s failed (%s: %s); retrying on %s",
                    name, use, type(e).__name__, str(e).splitlines()[0] if str(e) else "", CPU)
        session = ort.InferenceSession(model_path, sess_options=_session_options(CPU), providers=[CPU])

    ks = KavachSession(session, name, requested, log_latency=bool(rt.get("log_latency", False)))
    if ks.actual_provider != requested:
        log.warning("%s: requested %s but session is running on %s", name, requested, ks.actual_provider)
    else:
        log.info("%s: session on %s", name, ks.actual_provider)

    with _registry_lock:
        _registry[name] = ks
    return ks


def get_session(name: str) -> KavachSession | None:
    with _registry_lock:
        return _registry.get(name)


def all_stats() -> list[dict[str, Any]]:
    with _registry_lock:
        sessions = list(_registry.values())
        external = list(_external.items())
    out = [s.stats() for s in sessions]
    for name, fn in external:
        try:
            out.append({"name": name, **fn()})
        except Exception as e:
            log.warning("stats for %s failed: %s", name, type(e).__name__)
    return out


def clear_registry() -> None:
    with _registry_lock:
        _registry.clear()
        _external.clear()


def system_info() -> dict[str, Any]:
    installed = installed_ort_packages()
    builds = [p for p in installed if not (p.startswith("onnxruntime-qnn==") and not p.split("==")[1].startswith("1."))]
    if len(builds) > 1:                                  # onnxruntime-qnn 2.x is a plugin, not a second build
        log.warning("multiple onnxruntime packages installed (they conflict): %s", installed)
    return {
        "onnxruntime_version": ort.__version__,
        "onnxruntime_packages": installed,
        "available_providers": available_providers(),
        "plugin_eps": register_plugin_eps(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m models.runtime", description=__doc__.splitlines()[0])
    p.add_argument("--info", action="store_true", help="print onnxruntime / provider / platform info")
    p.add_argument("--model", help="ONNX model to benchmark")
    p.add_argument("--provider", choices=tuple(PROVIDER_NAMES), help="override config runtime.provider")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.info and not args.model:
        args.info = True

    if args.info:
        print(json.dumps(system_info(), indent=2))

    if args.model:
        sess = create_session(args.model, Path(args.model).stem, provider=args.provider)
        print(f"inputs:  {sess.input_specs()}")
        print(f"outputs: {sess.output_specs()}")
        sess.warmup(args.warmup)
        sess.reset_stats()
        feeds = sess.dummy_feeds()
        for _ in range(args.runs):
            sess.run(feeds)
        s = sess.stats()
        print(f"requested provider: {s['requested_provider']}")
        print(f"actual provider:    {s['actual_provider']}")
        print(f"runs: {s['n']}  p50: {s['p50_ms']:.3f} ms  p95: {s['p95_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
