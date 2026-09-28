"""Offline-at-runtime benchmark helper: profile ONNX models per execution provider and summarise latencies.

Dev-time only. Compiles and profiles our static OCR models on Qualcomm AI Hub (real Snapdragon X
devices) with the qai-hub client. Kavach's runtime never imports this module and never talks to
the network.

PRIVACY: the only things this module uploads are ONNX model files from weights/ and synthetic test
inputs generated in memory (the "Enter OTP 482913" image). Never pass screenshots, OCR text,
transcripts or audio to any function here.

Setup (once):  qai-hub configure --api_token <token from aihub.qualcomm.com → Settings>
               (stored in %USERPROFILE%\\.qai_hub\\client.ini, outside the repo)

    python -m bench.aihub_profile --list-devices
    python -m bench.aihub_profile --submit ocr [--inference-check]   # compile jobs; returns at once
    python -m bench.aihub_profile --status      # job states; submits profile/inference jobs for
                                                # compile jobs that have finished (no waiting)
    python -m bench.aihub_profile --collect     # download finished profiles/outputs -> aihub_results.json

Job records are appended to bench/results/aihub_jobs.json (older entries are never rewritten).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "bench" / "results"
JOBS_FILE = RESULTS_DIR / "aihub_jobs.json"
RESULTS_FILE = RESULTS_DIR / "aihub_results.json"
OCR_DIR = ROOT / "weights" / "ocr"

TOKEN_HELP = ("AI Hub API token not configured. Get it from https://aihub.qualcomm.com (Settings) and run:\n"
              "    qai-hub configure --api_token <token>\n"
              "It is stored in %USERPROFILE%\\.qai_hub\\client.ini (outside the repo).")


def _hub():
    try:
        import qai_hub
    except ImportError:
        raise SystemExit("qai-hub is not installed: pip install qai-hub") from None
    return qai_hub


def _bench_cfg() -> dict:
    from kavach_config import get_config

    return dict(get_config().get("bench", {}))


# ---------------------------------------------------------------- job / result files

def load_json_list(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path} must contain a JSON list")
    return data


def append_records(path: Path, records: list[dict]) -> None:
    """Append-only: existing entries are kept as they are; the file is replaced atomically."""
    existing = load_json_list(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(existing + records, f, indent=2)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _record(kind: str, job, model: str, device: str, runtime: str, **extra) -> dict:
    return {"ts": _now(), "kind": kind, "job_id": job.job_id, "url": job.url, "model": model,
            "device": device, "target_runtime": runtime, **extra}


# ---------------------------------------------------------------- devices

def is_snapdragon_x(name: str, attributes: list[str]) -> bool:
    return "snapdragon x" in name.lower() or any(a.startswith("chipset:qualcomm-snapdragon-x") for a in attributes)


def list_devices() -> list:
    hub = _hub()
    devices = [d for d in hub.get_devices() if is_snapdragon_x(d.name, list(d.attributes))]
    seen = set()
    for d in sorted(devices, key=lambda d: d.name):
        if d.name in seen:
            continue
        seen.add(d.name)
        attrs = [a for a in d.attributes if a.split(":")[0] in ("chipset", "os", "framework", "format", "vendor")]
        print(f"{d.name}  (os {d.os})\n    {', '.join(attrs)}")
    cfg = _bench_cfg()
    want = [cfg.get("aihub_device"), *cfg.get("aihub_devices_extra", [])]
    for name in filter(None, want):
        print(f"config device {name!r}: {'available' if name in seen else 'NOT FOUND in AI Hub device list'}")
    return devices


def configured_devices() -> list[str]:
    cfg = _bench_cfg()
    names = [cfg.get("aihub_device"), *cfg.get("aihub_devices_extra", [])]
    names = [n for n in names if n]
    if not names:
        raise SystemExit("set bench.aihub_device in config.yaml (see --list-devices)")
    return names


# ---------------------------------------------------------------- models

def ocr_models() -> list[Path]:
    from kavach_config import get_config

    buckets = sorted(int(w) for w in get_config().ocr.rec_buckets)
    paths = [OCR_DIR / "det_static.onnx", *(OCR_DIR / f"rec_static_{w}.onnx" for w in buckets)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing {missing}; run scripts/fetch_ocr_models.py and scripts/fix_ocr_shapes.py")
    return paths


def onnx_input_specs(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Fixed input specs read from the ONNX file ({name: (shape, dtype)}); every dim must be static."""
    import onnx

    elem = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.INT32: "int32", onnx.TensorProto.INT64: "int64",
            onnx.TensorProto.UINT8: "uint8", onnx.TensorProto.INT8: "int8", onnx.TensorProto.FLOAT16: "float16"}
    model = onnx.load(str(path), load_external_data=False)
    inits = {i.name for i in model.graph.initializer}
    specs = {}
    for vi in model.graph.input:
        if vi.name in inits:
            continue
        dims = [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]
        if any(d is None or d <= 0 for d in dims):
            raise SystemExit(f"{path.name}: input {vi.name} is not static {dims}; run scripts/fix_ocr_shapes.py")
        specs[vi.name] = (tuple(dims), elem.get(vi.type.tensor_type.elem_type, "float32"))
    return specs


# ---------------------------------------------------------------- submit

def _submit_compile(hub, model_path: Path, device, specs, runtimes: list[str]):
    """Try each target runtime in order; return (job, runtime) for the first the server accepts."""
    last_err = None
    for rt in runtimes:
        try:
            job = hub.submit_compile_job(model=str(model_path), device=device, name=f"kavach {model_path.stem}",
                                         input_specs=specs, options=f"--target_runtime {rt}")
            return job, rt
        except hub.UserError as e:  # e.g. unknown/unsupported target runtime
            print(f"  {model_path.name}: --target_runtime {rt} rejected: {e}")
            last_err = e
    raise SystemExit(f"{model_path.name}: no target runtime accepted ({last_err})")


def submit_ocr(inference_check: bool = False) -> list[dict]:
    hub = _hub()
    cfg = _bench_cfg()
    runtimes = [cfg.get("aihub_target_runtime", "precompiled_qnn_onnx")]
    if cfg.get("aihub_fallback_runtime") and cfg["aihub_fallback_runtime"] not in runtimes:
        runtimes.append(cfg["aihub_fallback_runtime"])
    records = []
    for dev_name in configured_devices():
        device = hub.Device(dev_name)
        for path in ocr_models():
            specs = onnx_input_specs(path)
            job, rt = _submit_compile(hub, path, device, specs, runtimes)
            rec = _record("compile", job, path.stem, dev_name, rt, model_file=str(path.relative_to(ROOT)),
                          model_sha256=_sha256(path), input_specs={k: list(v[0]) for k, v in specs.items()},
                          wants=["profile"] + (["inference"] if inference_check else []))
            records.append(rec)
            append_records(JOBS_FILE, [rec])  # save immediately, one job at a time
            print(f"compile  {path.stem:<18} {dev_name:<26} {rt:<22} {job.job_id}  {job.url}")
    print(f"\n{len(records)} compile job(s) submitted; not waiting. Profile"
          f"{'/inference' if inference_check else ''} jobs are submitted by --status once compiles finish.")
    return records


# ---------------------------------------------------------------- synthetic inputs (inference check)

def synthetic_inputs(model_stem: str) -> dict[str, np.ndarray]:
    """One real preprocessed input for a static model, from the synthetic OTP image (in memory only)."""
    from models import ocr_native as N
    from models.ocr import synthetic_image

    img, _ = N.global_preprocess(synthetic_image())
    if model_stem == "det_static":
        specs = onnx_input_specs(OCR_DIR / "det_static.onnx")
        (name, ((_, _, h, w), _)), = specs.items()
        tensor, _ = N.letterbox_det(img, h, w)
        return {name: tensor}
    if model_stem.startswith("rec_static_"):
        specs = onnx_input_specs(OCR_DIR / f"{model_stem}.onnx")
        (name, ((b, _, h, w), _)), = specs.items()
        crop = _synthetic_crop(img)
        batch = np.zeros((b, 3, h, w), np.float32)
        batch[0] = N.normalize_rec(crop, w, h)  # row 0 = the real text line; other rows = padding
        return {name: batch}
    raise ValueError(f"no synthetic input for {model_stem}")


def _synthetic_crop(img: np.ndarray) -> np.ndarray:
    """The text-line crop of the synthetic image, found with the local CPU det model."""
    from models import ocr_native as N
    from models import runtime

    sess = runtime.create_session(OCR_DIR / "det_static.onnx", "aihub.det.local", provider="cpu")
    _, _, h, w = sess.input_specs()[0]["shape"]
    tensor, (vh, vw) = N.letterbox_det(img, h, w)
    pred = sess.run({sess.input_specs()[0]["name"]: tensor})[0][:, :, :vh, :vw]
    boxes, _ = N.DBPostProcess()(pred, img.shape[:2])
    boxes = N.filter_tag_det_res(boxes, img.shape[:2])
    if not len(boxes):
        raise SystemExit("local det found no text in the synthetic image")
    return N.get_rotate_crop_image(img, N.sorted_boxes(boxes)[0].copy())


def local_cpu_outputs(model_stem: str, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    from models import runtime

    sess = runtime.create_session(OCR_DIR / f"{model_stem}.onnx", f"aihub.{model_stem}.local", provider="cpu")
    return sess.run(inputs)


# ---------------------------------------------------------------- status / advance

def _index(jobs: list[dict]) -> dict[str, list[dict]]:
    children: dict[str, list[dict]] = {}
    for j in jobs:
        if j.get("parent_job_id"):
            children.setdefault(j["parent_job_id"], []).append(j)
    return children


def status(advance: bool = True) -> None:
    hub = _hub()
    jobs = load_json_list(JOBS_FILE)
    if not jobs:
        print(f"no jobs recorded in {JOBS_FILE.relative_to(ROOT)}")
        return
    children = _index(jobs)
    new_records = []
    for j in jobs:
        job = hub.get_job(j["job_id"])
        st = job.get_status()
        msg = f"  -- {st.message}" if st.failure and st.message else ""
        print(f"{j['kind']:<9} {j['model']:<18} {j['device']:<26} {st.code:<22} {j['job_id']}{msg}")
        if not (advance and j["kind"] == "compile" and st.success):
            continue
        have = {c["kind"] for c in children.get(j["job_id"], [])}
        todo = [k for k in j.get("wants", ["profile"]) if k not in have]
        if not todo:
            continue
        target = job.get_target_model()  # compile already finished: returns immediately
        device = hub.Device(j["device"])
        if "profile" in todo:
            pj = hub.submit_profile_job(model=target, device=device, name=f"kavach {j['model']}")
            new_records.append(_record("profile", pj, j["model"], j["device"], j["target_runtime"],
                                       parent_job_id=j["job_id"]))
            print(f"  -> submitted profile job {pj.job_id}  {pj.url}")
        if "inference" in todo:
            inputs = synthetic_inputs(j["model"])  # synthetic only (privacy)
            ij = hub.submit_inference_job(model=target, device=device, name=f"kavach {j['model']} check",
                                          inputs={k: [v] for k, v in inputs.items()})
            new_records.append(_record("inference", ij, j["model"], j["device"], j["target_runtime"],
                                       parent_job_id=j["job_id"]))
            print(f"  -> submitted inference job {ij.job_id}  {ij.url}")
    if new_records:
        append_records(JOBS_FILE, new_records)


# ---------------------------------------------------------------- collect

def summarize_profile(profile: dict) -> dict[str, Any]:
    """Estimated time (ms), peak memory (MB) and ops per compute unit from a downloaded profile."""
    summ = profile.get("execution_summary", {})
    t_us, mem_b = summ.get("estimated_inference_time"), summ.get("estimated_inference_peak_memory")
    units = Counter(str(d.get("compute_unit", "UNKNOWN")).upper() for d in profile.get("execution_detail", []))
    return {
        "inference_ms": round(t_us / 1000, 3) if isinstance(t_us, (int, float)) and t_us >= 0 else None,
        "peak_memory_mb": round(mem_b / 1024 / 1024, 1) if isinstance(mem_b, (int, float)) and mem_b >= 0 else None,
        "ops": {u: units.get(u, 0) for u in ("NPU", "GPU", "CPU")} | {k: v for k, v in units.items()
                                                                        if k not in ("NPU", "GPU", "CPU")},
    }


def compare_outputs(model_stem: str, remote: dict, local: list[np.ndarray]) -> dict[str, Any]:
    remote_arrays = [np.asarray(v[0] if isinstance(v, list) else v) for v in remote.values()]
    out: dict[str, Any] = {"max_abs_diff": [float(np.max(np.abs(r.astype(np.float32) - l)))
                                            for r, l in zip(remote_arrays, local)]}
    if model_stem.startswith("rec_static_"):
        from models import ocr_native as N
        from kavach_config import get_config

        chars = N.load_keys(ROOT / get_config().ocr.get("keys", "weights/ocr/keys.txt"))
        (r_text, _), = N.ctc_decode(remote_arrays[0][:1], chars)
        (l_text, _), = N.ctc_decode(local[0][:1], chars)
        # The decoded text is the synthetic "Enter OTP 482913" line, not user data.
        out |= {"text_equal": r_text == l_text, "remote_text": r_text, "local_text": l_text}
    return out


def collect() -> None:
    hub = _hub()
    jobs = load_json_list(JOBS_FILE)
    done = {r["job_id"] for r in load_json_list(RESULTS_FILE)}
    new = []
    for j in jobs:
        if j["kind"] not in ("profile", "inference") or j["job_id"] in done:
            continue
        job = hub.get_job(j["job_id"])
        st = job.get_status()
        if st.failure:
            print(f"FAILED   {j['kind']:<9} {j['model']:<18} {j['device']}  {j['job_id']}\n         reason: {st.message}")
            new.append({"ts": _now(), "job_id": j["job_id"], "kind": j["kind"], "model": j["model"],
                        "device": j["device"], "status": "FAILED", "failure_reason": st.message})
            continue
        if not st.success:
            print(f"pending  {j['kind']:<9} {j['model']:<18} {j['device']}  {st.code}")
            continue
        base = {"ts": _now(), "job_id": j["job_id"], "kind": j["kind"], "model": j["model"], "device": j["device"],
                "target_runtime": j["target_runtime"], "url": j["url"], "status": "SUCCESS"}
        if j["kind"] == "profile":
            s = summarize_profile(job.download_profile())
            print(f"profile  {j['model']:<18} {j['device']:<26} {s['inference_ms']} ms  "
                  f"peak {s['peak_memory_mb']} MB  ops {s['ops']}")
            new.append(base | s)
        else:
            inputs = synthetic_inputs(j["model"])
            cmp = compare_outputs(j["model"], job.download_output_data(), local_cpu_outputs(j["model"], inputs))
            print(f"infer    {j['model']:<18} {j['device']:<26} max|diff| {cmp['max_abs_diff']}"
                  + (f"  text equal: {cmp['text_equal']}" if "text_equal" in cmp else ""))
            new.append(base | cmp)
    if new:
        append_records(RESULTS_FILE, new)
        print(f"\nsaved {len(new)} result(s) to {RESULTS_FILE.relative_to(ROOT)}")
    else:
        print("nothing new to collect")


# ---------------------------------------------------------------- CLI

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.aihub_profile", description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-devices", action="store_true")
    g.add_argument("--submit", choices=["ocr"])
    g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true")
    p.add_argument("--inference-check", action="store_true",
                   help="with --submit: also run one synthetic input on device and compare to local CPU")
    p.add_argument("--no-advance", action="store_true", help="with --status: only print, submit nothing")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    hub = _hub()
    try:
        if args.list_devices:
            list_devices()
        elif args.submit:
            submit_ocr(args.inference_check)
        elif args.status:
            status(advance=not args.no_advance)
        else:
            collect()
    except hub.UserError as e:
        msg = str(e).lower()
        if any(k in msg for k in ("client.ini", "configuration file", "api_token", "api key", "token")):
            print(TOKEN_HELP)
            return 2
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
