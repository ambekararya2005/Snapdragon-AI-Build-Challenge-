"""Dev-time: download the AI Hub compiled NPU models (precompiled_qnn_onnx) into weights/qnn/.

Reads the finished compile jobs from bench/results/aihub_jobs.json (det_static, rec_static_<W>,
whisper_base_encoder / _decoder), downloads each job's target model, and writes weights/qnn/manifest.json
with the source job ids, device, runtime, file sizes and sha256, plus each model's graph inputs/outputs.

Layout (one folder per model: a precompiled ONNX's EPContext node refers to its QNN context binary by
file name, so the files stay together and keep their names; only the .onnx is renamed):
    weights/qnn/det_static/det_static.onnx (+ context .bin)
    weights/qnn/rec_static_<W>/rec_static_<W>.onnx (+ .bin)
    weights/qnn/whisper_base/encoder/encoder.onnx, decoder/decoder.onnx (+ .bin),
        whisper.json / mel_filters.npy / tokens.json copied from weights/asr/whisper_base with io files
        pointed at the compiled models
config.snapdragon.yaml points the native OCR backend and the ASR at these files.

Network: talks to Qualcomm AI Hub (needs `qai-hub configure --api_token ...`; the token stays in
%USERPROFILE%\\.qai_hub\\client.ini, outside the repo). Dev-time only; the Kavach runtime never downloads.

    python scripts\\fetch_qnn_models.py            # download all, write the manifest
    python scripts\\fetch_qnn_models.py --check    # print what is on disk vs the manifest, no network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "bench" / "results" / "aihub_jobs.json"
OUT = ROOT / "weights" / "qnn"
ASR_SRC = ROOT / "weights" / "asr" / "whisper_base"
RUNTIME = "precompiled_qnn_onnx"
WHISPER = {"whisper_base_encoder": "encoder", "whisper_base_decoder": "decoder"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compile_jobs() -> dict[str, dict]:
    """Latest compile job per model for the precompiled_qnn_onnx runtime."""
    jobs = json.loads(JOBS.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for j in jobs:
        if j.get("kind") == "compile" and j.get("target_runtime") == RUNTIME:
            out[j["model"]] = j                     # later entries win
    return out


def target_dir(model: str) -> Path:
    if model in WHISPER:
        return OUT / "whisper_base" / WHISPER[model]
    return OUT / model


def onnx_name(model: str) -> str:
    return f"{WHISPER.get(model, model)}.onnx"


def graph_io(path: Path) -> dict:
    """Graph inputs/outputs (name, dims, elem type) without loading external data."""
    import onnx

    m = onnx.load(str(path), load_external_data=False)

    def spec(v):
        t = v.type.tensor_type
        return [v.name, [d.dim_value or d.dim_param for d in t.shape.dim], onnx.TensorProto.DataType.Name(t.elem_type)]

    ops = sorted({n.op_type for n in m.graph.node})
    return {"inputs": [spec(v) for v in m.graph.input], "outputs": [spec(v) for v in m.graph.output], "ops": ops}


def unpack(downloaded: Path, dest: Path, model: str) -> Path:
    """Move a downloaded model (file, directory or zip) into dest; returns the .onnx path (renamed)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if downloaded.is_file() and zipfile.is_zipfile(downloaded):
        with zipfile.ZipFile(downloaded) as z:
            z.extractall(dest)
    elif downloaded.is_dir():
        shutil.copytree(downloaded, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(downloaded, dest / downloaded.name)
    # flatten a single wrapping folder (zip / dir with one top-level dir)
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for p in inner.iterdir():
            shutil.move(str(p), dest / p.name)
        inner.rmdir()
    onnx_files = sorted(dest.rglob("*.onnx"))
    if len(onnx_files) != 1:
        raise RuntimeError(f"{model}: expected one .onnx in the download, found {[p.name for p in onnx_files]}")
    target = dest / onnx_name(model)
    if onnx_files[0] != target:
        onnx_files[0].rename(target)
    return target


def write_whisper_meta(manifest_models: dict) -> None:
    """whisper.json / mel_filters.npy / tokens.json next to the compiled encoder/decoder, io files updated."""
    wdir = OUT / "whisper_base"
    for name in ("mel_filters.npy", "tokens.json"):
        shutil.copy2(ASR_SRC / name, wdir / name)
    meta = json.loads((ASR_SRC / "whisper.json").read_text(encoding="utf-8"))
    for model, part in WHISPER.items():
        meta["io"][part]["file"] = f"{part}/{onnx_name(model)}"
        io = manifest_models[model]["graph"]
        want_in = set(meta["io"][part]["inputs"])
        want_out = set(meta["io"][part]["outputs"])
        got_in = {i[0] for i in io["inputs"]}
        got_out = {o[0] for o in io["outputs"]}
        manifest_models[model]["io_matches_plain_export"] = {"inputs": got_in == want_in, "outputs": got_out == want_out}
    meta["compiled_for"] = "QNN HTP (AI Hub precompiled_qnn_onnx); see weights/qnn/manifest.json"
    (wdir / "whisper.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def fetch() -> int:
    import qai_hub as hub

    jobs = compile_jobs()
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runtime": RUNTIME,
                "source": "bench/results/aihub_jobs.json (compile jobs)", "models": {}}
    failed = []
    for model, j in sorted(jobs.items()):
        job = hub.get_job(j["job_id"])
        st = job.get_status()
        print(f"{model:<22} job {j['job_id']}  status {st.code}", flush=True)
        if not st.success:
            failed.append(model)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(job.get_target_model().download(str(Path(tmp) / model)))
            onnx_path = unpack(downloaded, target_dir(model), model)
        files = sorted(p for p in target_dir(model).rglob("*") if p.is_file())
        manifest["models"][model] = {
            "compile_job_id": j["job_id"], "job_url": j.get("url"), "device": j.get("device"),
            "source_model_sha256": j.get("model_sha256"), "onnx": onnx_path.relative_to(ROOT).as_posix(),
            "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
            "graph": graph_io(onnx_path),
        }
        print(f"    -> {onnx_path.relative_to(ROOT)} ({len(files)} files, "
              f"{sum(p.stat().st_size for p in files) / 1e6:.1f} MB)", flush=True)
    if all(m in manifest["models"] for m in WHISPER):
        write_whisper_meta(manifest["models"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {(OUT / 'manifest.json').relative_to(ROOT)}; failed: {failed or 'none'}")
    return 1 if failed else 0


def check() -> int:
    path = OUT / "manifest.json"
    if not path.is_file():
        print("no weights/qnn/manifest.json; run without --check first")
        return 1
    manifest = json.loads(path.read_text(encoding="utf-8"))
    bad = 0
    for model, m in manifest["models"].items():
        for rel, f in m["files"].items():
            p = ROOT / rel
            ok = p.is_file() and p.stat().st_size == f["bytes"] and sha256(p) == f["sha256"]
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {model:<22} {rel}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download AI Hub precompiled QNN models into weights/qnn/")
    p.add_argument("--check", action="store_true", help="verify files against the manifest (no network)")
    args = p.parse_args(argv)
    return check() if args.check else fetch()


if __name__ == "__main__":
    sys.exit(main())
