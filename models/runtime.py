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
from typing import Any, Mapping, Sequence

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


def _session_options(provider_name: str) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider_name == PROVIDER_NAMES["dml"]:
        # DirectML does not support memory pattern optimisation or parallel execution.
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def _providers_list(provider_name: str, options: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    if provider_name == CPU:
        return [(CPU, {})]
    return [(provider_name, options), (CPU, {})]


class KavachSession:
    """Wraps an onnxruntime.InferenceSession and records per-run latency."""

    def __init__(self, session: ort.InferenceSession, name: str, requested_provider: str, log_latency: bool = False):
        self.session = session
        self.name = name
        self.requested_provider = requested_provider
        self.actual_provider = session.get_providers()[0]
        self.log_latency = log_latency
        self._lat_ms: deque[float] = deque(maxlen=STATS_WINDOW)
        self._n = 0
        self._lock = threading.Lock()

    def run(self, feeds: Mapping[str, np.ndarray], output_names: Sequence[str] | None = None) -> list[np.ndarray]:
        t0 = time.perf_counter()
        outputs = self.session.run(list(output_names) if output_names else None, dict(feeds))
        ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._lat_ms.append(ms)
            self._n += 1
        if self.log_latency:
            log.debug("%s: run %.2f ms on %s", self.name, ms, self.actual_provider)
        return outputs

    def input_specs(self) -> list[dict[str, Any]]:
        return [{"name": i.name, "shape": list(i.shape), "dtype": i.type} for i in self.session.get_inputs()]

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
        with self._lock:
            self._lat_ms.clear()
            self._n = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            lat = np.array(self._lat_ms, dtype=np.float64)
            n = self._n
        return {
            "name": self.name,
            "requested_provider": self.requested_provider,
            "actual_provider": self.actual_provider,
            "n": n,
            "last_ms": round(float(lat[-1]), 3) if lat.size else None,
            "p50_ms": round(float(np.percentile(lat, 50)), 3) if lat.size else None,
            "p95_ms": round(float(np.percentile(lat, 95)), 3) if lat.size else None,
        }


_registry: dict[str, KavachSession] = {}
_registry_lock = threading.Lock()


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

    available = ort.get_available_providers()
    use = requested
    if requested not in available:
        if not fallback:
            raise RuntimeError(f"{name}: requested {requested} is not available (available: {available})")
        log.warning("%s: requested %s is not available (available: %s); using %s", name, requested, available, CPU)
        use = CPU

    try:
        session = ort.InferenceSession(
            model_path, sess_options=_session_options(use), providers=_providers_list(use, options)
        )
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
    return [s.stats() for s in sessions]


def clear_registry() -> None:
    with _registry_lock:
        _registry.clear()


def system_info() -> dict[str, Any]:
    installed = []
    for pkg in ORT_PACKAGES:
        try:
            installed.append(f"{pkg}=={importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) > 1:
        log.warning("multiple onnxruntime packages installed (they conflict): %s", installed)
    return {
        "onnxruntime_version": ort.__version__,
        "onnxruntime_packages": installed,
        "available_providers": ort.get_available_providers(),
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
