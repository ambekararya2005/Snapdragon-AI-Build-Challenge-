"""Dev-time benchmark: the same ONNX files on the dev PC (CPU, RTX 4050 via DirectML) through
models.runtime, merged with the AI Hub (Snapdragon X Elite NPU) profiles into
bench/results/benchmark_table.md (TRD section 15 format).

Models: OCR det + rec for both backends (native static models = the NPU path; rapidocr dynamic models
run at the same input shapes), Whisper-Base encoder and one decoder step. Plus end-to-end rows: OCR on
a synthetic 1280x720 frame (both backends) and Whisper per 5 s chunk computed from measured parts.

Every model input is synthetic (zeros / a rendered test frame). Values that were not measured are shown
as "pending" or with the AI Hub job status - never estimated. Raw numbers: bench/results/local_bench.json.

    python -m bench.local_bench                # collect AI Hub results, benchmark, write the table
    python -m bench.local_bench --runs 50 --no-collect
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.aihub_profile import JOBS_FILE, RESULTS_FILE, load_json_list  # noqa: E402
from kavach_config import get_config  # noqa: E402
from models import runtime  # noqa: E402

RESULTS_DIR = ROOT / "bench" / "results"
TABLE_FILE = RESULTS_DIR / "benchmark_table.md"
RAW_FILE = RESULTS_DIR / "local_bench.json"
PROVIDERS = ("cpu", "dml")
WHISPER = "weights/asr/whisper_base"
DECODER_STEPS = 20        # 4 forced prefix tokens + 16 text tokens (a 5 s speech chunk decoded 9-17 live)
NO_NPU = "n/a - dynamic shapes, not compiled for the NPU (native backend is the NPU path)"

# (row, backend, onnx path, input shape override or None, AI Hub model name or None)
MODELS = [
    ("OCR det (1x3x736x1280)", "native", "weights/ocr/det_static.onnx", None, "det_static"),
    ("OCR det (1x3x736x1280)", "rapidocr", "weights/ocr/det.onnx", (1, 3, 736, 1280), None),
    ("OCR rec, batch 8x3x48x320", "native", "weights/ocr/rec_static_320.onnx", None, "rec_static_320"),
    ("OCR rec, batch 8x3x48x320", "rapidocr", "weights/ocr/rec.onnx", (8, 3, 48, 320), None),
    ("OCR rec, batch 8x3x48x640", "native", "weights/ocr/rec_static_640.onnx", None, "rec_static_640"),
    ("OCR rec, batch 8x3x48x640", "rapidocr", "weights/ocr/rec.onnx", (8, 3, 48, 640), None),
    ("OCR rec, batch 4x3x48x1280", "native", "weights/ocr/rec_static_1280.onnx", None, "rec_static_1280"),
    ("OCR rec, batch 4x3x48x1280", "rapidocr", "weights/ocr/rec.onnx", (4, 3, 48, 1280), None),
    ("Whisper-Base encoder (1x80x3000 = 30 s)", "aihub_whisper", f"{WHISPER}/encoder.onnx", None,
     "whisper_base_encoder"),
    ("Whisper-Base decoder, 1 step (200-slot KV cache)", "aihub_whisper", f"{WHISPER}/decoder.onnx", None,
     "whisper_base_decoder"),
]


# ---------------------------------------------------------------- local timing

def _pct(ms: list[float]) -> dict[str, float]:
    return {"p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2),
            "n": len(ms)}


def time_model(path: str, provider: str, shape: tuple | None, runs: int, warmup: int) -> dict[str, Any]:
    rt = {**get_config().runtime.to_dict(), "provider": provider, "fallback_to_cpu": False, "log_latency": False}
    try:
        ks = runtime.create_session(ROOT / path, f"bench:{Path(path).stem}:{provider}", provider, rt)
        name = ks.input_specs()[0]["name"]
        feeds = ks.dummy_feeds({name: shape} if shape else None)
        for _ in range(warmup):
            ks.run(feeds)
        ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            ks.run(feeds)
            ms.append((time.perf_counter() - t0) * 1000.0)
        return {**_pct(ms), "provider": ks.actual_provider}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"}
    finally:
        runtime.clear_registry()


def synthetic_frame() -> np.ndarray:
    import cv2
    img = np.full((720, 1280, 3), 255, np.uint8)
    lines = ["Net Banking - Fund Transfer", "Beneficiary: DEMO NOT A REAL BANK", "IFSC DEMO0001234",
             "Amount Rs 49,999.00", "Enter OTP sent to your mobile", "Do not share your OTP with anyone"]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (40, 80 + i * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return img


def time_ocr_frame(backend: str, provider: str, runs: int, warmup: int) -> dict[str, Any]:
    from models.ocr import create_backend
    cfg = get_config().to_dict()
    cfg["ocr"]["backend"] = backend
    cfg["runtime"].update(provider=provider, fallback_to_cpu=False, log_latency=False)
    try:
        ocr = create_backend(cfg)
        img = synthetic_frame()
        for _ in range(warmup):
            ocr(img)
        ms, n_lines = [], 0
        for _ in range(runs):
            r = ocr(img)
            ms.append(r.total_ms)
            n_lines = len(r.lines)
        return {**_pct(ms), "provider": ocr.provider, "lines": n_lines}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:160]}
    finally:
        runtime.clear_registry()


# ---------------------------------------------------------------- AI Hub

def aihub_rows() -> dict[str, dict[str, Any]]:
    """Latest profile per AI Hub model: result if collected, else the live job status."""
    jobs = [j for j in load_json_list(JOBS_FILE) if j["kind"] == "profile"]
    results = {r["job_id"]: r for r in load_json_list(RESULTS_FILE) if r.get("kind") == "profile"}
    out: dict[str, dict[str, Any]] = {}
    for j in jobs:                                         # later records win
        r = results.get(j["job_id"])
        if r is None:
            try:
                import qai_hub as hub
                st = hub.get_job(j["job_id"]).get_status()
                r = {"status": "FAILED" if st.failure else f"pending ({st.code})", "failure_reason": st.message}
            except Exception as e:  # noqa: BLE001
                r = {"status": "pending (status unknown)", "failure_reason": type(e).__name__}
        out[j["model"]] = {**r, "job_id": j["job_id"], "device": j["device"], "target_runtime": j["target_runtime"]}
    return out


def aihub_tool_versions(job_id: str) -> str:
    try:
        import qai_hub as hub
        from qai_hub.public_rest_api import get_job_results
        job = hub.get_job(job_id)
        tv = get_job_results(job._owner.config, job_id).profile_job_result.profile.tool_versions
        return ", ".join(f"{t.name} {t.version}" for t in tv) or "unknown"
    except Exception as e:  # noqa: BLE001
        return f"unknown ({type(e).__name__})"


# ---------------------------------------------------------------- environment

def cpu_name() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:  # noqa: BLE001
        return platform.processor() or "unknown CPU"


def nvidia_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown (nvidia-smi unavailable)"


# ---------------------------------------------------------------- table

def _cell(m: dict | None) -> str:
    if not m:
        return "pending"
    if "error" in m:
        return f"failed: {m['error']}"
    return f"{m['p50']:.1f} / {m['p95']:.1f}"


def _npu_cell(hub_row: dict | None, model: str | None) -> tuple[str, str, str]:
    if model is None:
        return NO_NPU, "-", "-"
    if hub_row is None:
        return "pending (no AI Hub job recorded)", "-", "-"
    if hub_row.get("status") != "SUCCESS":
        reason = f": {hub_row['failure_reason']}" if hub_row.get("failure_reason") else ""
        return f"{hub_row['status']}{reason} ({hub_row['job_id']})", "-", "-"
    ops = hub_row.get("ops") or {}
    split = " / ".join(str(ops.get(k, 0)) for k in ("NPU", "GPU", "CPU"))
    return f"{hub_row['inference_ms']:.1f}", split, f"{hub_row['peak_memory_mb']:.1f} MB"


def write_table(raw: dict[str, Any]) -> str:
    env, hub_rows = raw["env"], raw["aihub"]
    local = {(r["row"], r["backend"]): r for r in raw["models"]}
    L = [f"# Kavach benchmark table", "",
         f"Generated {raw['date']} by `python -m bench.local_bench` ({raw['runs']} timed runs after {raw['warmup']} "
         "warm-up runs per model; synthetic inputs only).", "",
         "| | Device | Runtime |", "|---|---|---|",
         f"| Snapdragon NPU (AI Hub) | {env['aihub_device']} (Hexagon NPU), AI Hub hosted | {env['aihub_runtime']}; "
         f"target `precompiled_qnn_onnx` |",
         f"| Dev CPU | {env['cpu']} | onnxruntime {env['ort_version']} ({', '.join(env['ort_packages'])}), "
         "CPUExecutionProvider |",
         f"| RTX 4050 (DirectML) | {env['gpu']} (DirectML adapter `dml.device_id: {env['dml_device_id']}`) | "
         f"onnxruntime {env['ort_version']}, DmlExecutionProvider |", "",
         "Local cells are **p50 / p95 ms**. The AI Hub cell is the profiler's estimated inference time (ms). "
         "Op split and peak memory come from the AI Hub profile. Radeon iGPU (DirectML device 0) numbers from "
         "earlier runs are not included.", "",
         "## Per model", "",
         "| Model | Backend | Snapdragon NPU (AI Hub) | Dev CPU | RTX 4050 (DirectML) | NPU / GPU / CPU ops (AI Hub) "
         "| Peak memory (AI Hub) |",
         "|---|---|---|---|---|---|---|"]
    for row, backend, _path, _shape, model in MODELS:
        r = local.get((row, backend), {})
        npu, split, mem = _npu_cell(hub_rows.get(model) if model else None, model)
        L.append(f"| {row} | {backend} | {npu} | {_cell(r.get('cpu'))} | {_cell(r.get('dml'))} | {split} | {mem} |")

    L += ["", "## End to end", "",
          "| Pipeline | Backend | Snapdragon NPU (AI Hub) | Dev CPU | RTX 4050 (DirectML) | Notes |",
          "|---|---|---|---|---|---|"]
    for f in raw["ocr_frames"]:
        lines = next((v["lines"] for v in (f.get("cpu"), f.get("dml")) if v and "lines" in v), "?")
        L.append(f"| OCR full frame, synthetic 1280x720 ({lines} lines) | {f['backend']} | pending (not measured "
                 f"end to end on device) | {_cell(f.get('cpu'))} | {_cell(f.get('dml'))} | det + rec + pre/post "
                 "processing, measured |")
    enc = local.get((MODELS[8][0], "aihub_whisper"), {})
    dec = local.get((MODELS[9][0], "aihub_whisper"), {})
    he, hd = hub_rows.get("whisper_base_encoder"), hub_rows.get("whisper_base_decoder")

    def comp(e: dict | None, d: dict | None) -> str:
        if not e or not d or "p50" not in e or "p50" not in d:
            return "pending"
        return f"{e['p50'] + DECODER_STEPS * d['p50']:.1f} (computed from measured parts)"

    if he and hd and he.get("status") == hd.get("status") == "SUCCESS":
        npu = (f"{he['inference_ms'] + DECODER_STEPS * hd['inference_ms']:.1f} (computed from AI Hub-measured "
               f"parts: {he['inference_ms']:.1f} + {DECODER_STEPS} x {hd['inference_ms']:.3f})")
    else:
        npu = "pending"
    L.append(f"| Whisper-Base per 5 s chunk = encoder + {DECODER_STEPS} decoder steps | aihub_whisper | {npu} | "
             f"{comp(enc.get('cpu'), dec.get('cpu'))} | {comp(enc.get('dml'), dec.get('dml'))} | p50 parts; excludes "
             "log-mel (~5 ms numpy) and host overhead; real-time factor = value / 5000 ms |")
    L += ["", "## AI Hub jobs", "", "| Model | Profile job | Status |", "|---|---|---|"]
    for model, h in sorted(hub_rows.items()):
        L.append(f"| {model} | [{h['job_id']}](https://workbench.aihub.qualcomm.com/jobs/{h['job_id']}/) | "
                 f"{h.get('status')} |")
    text = "\n".join(L) + "\n"
    TABLE_FILE.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.local_bench", description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-collect", action="store_true", help="skip python -m bench.aihub_profile --collect")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if not args.no_collect:
        from bench import aihub_profile
        print("== AI Hub collect ==")
        try:
            aihub_profile.collect()
        except SystemExit as e:
            print(f"collect skipped: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"collect failed: {type(e).__name__}: {e}")

    info = runtime.system_info()
    hub_rows = aihub_rows()
    first_ok = next((h["job_id"] for h in hub_rows.values() if h.get("status") == "SUCCESS"), None)
    raw: dict[str, Any] = {
        "date": time.strftime("%Y-%m-%d %H:%M %z"), "runs": args.runs, "warmup": args.warmup,
        "env": {"cpu": cpu_name(), "gpu": nvidia_name(), "dml_device_id": get_config().runtime.dml.device_id,
                "ort_version": info["onnxruntime_version"], "ort_packages": info["onnxruntime_packages"],
                "aihub_device": get_config().bench.aihub_device,
                "aihub_runtime": aihub_tool_versions(first_ok) if first_ok else "unknown"},
        "aihub": hub_rows, "models": [], "ocr_frames": [],
    }
    print(f"\n== local benchmark: {args.runs} runs after {args.warmup} warm-up ==")
    for row, backend, path, shape, _model in MODELS:
        rec = {"row": row, "backend": backend, "path": path}
        for prov in PROVIDERS:
            rec[prov] = time_model(path, prov, shape, args.runs, args.warmup)
        raw["models"].append(rec)
        print(f"{row:<50} {backend:<14} cpu {_cell(rec['cpu']):<16} dml {_cell(rec['dml'])}")
    for backend in ("native", "rapidocr"):
        rec = {"backend": backend}
        for prov in PROVIDERS:
            rec[prov] = time_ocr_frame(backend, prov, max(10, args.runs // 2), args.warmup)
        raw["ocr_frames"].append(rec)
        print(f"{'OCR full frame (synthetic 1280x720)':<50} {backend:<14} cpu {_cell(rec['cpu']):<16} "
              f"dml {_cell(rec['dml'])}")
    RAW_FILE.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    table = write_table(raw)
    print(f"\n== {TABLE_FILE.relative_to(ROOT)} ==\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
