"""Dev-time: export a qai_hub_models Whisper to plain ONNX + runtime assets, and submit AI Hub jobs.

Runs ONLY in the separate export venv (.venv-aihub: torch, transformers, qai-hub-models). The Kavach
runtime (.venv) needs none of those: models/asr.py uses the files written here with numpy + onnxruntime.

Writes to weights/asr/<model>/:
    encoder.onnx, decoder.onnx   plain static ONNX (CPU/DML/QNN-EP-capable), traced from the same
                                 patched torch modules qai_hub_models compiles for AI Hub
    mel_filters.npy              (201, 80) Slaney mel filterbank from WhisperFeatureExtractor
    tokens.json                  token id -> hex bytes (byte-level BPE decoded), special ids
    whisper.json                 fixed shapes, special token ids, suppress lists, I/O names

AI Hub (--aihub): runs the model's own export pipeline (python -m qai_hub_models.models.<model>.export)
for bench.aihub_device with precompiled_qnn_onnx; compile/link/profile job ids are appended to
bench/results/aihub_jobs.json. Profiles are then collected with `python -m bench.aihub_profile --collect`.
PRIVACY: only model weights and the package's own sample inputs are uploaded.

Setup (once):
    py -3.11 -m venv .venv-aihub
    .\\.venv-aihub\\Scripts\\python.exe -m pip install "qai-hub-models[whisper-base]"

    .\\.venv-aihub\\Scripts\\python.exe scripts\\export_whisper.py --model whisper_base
    .\\.venv-aihub\\Scripts\\python.exe scripts\\export_whisper.py --model whisper_base --skip-onnx --aihub
    .\\.venv-aihub\\Scripts\\python.exe scripts\\export_whisper.py --faster-whisper base.en   # dev fallback weights
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _model(name: str):
    return importlib.import_module(f"qai_hub_models.models.{name}").Model.from_pretrained()


def _zeros(spec) -> tuple:
    import torch
    return tuple(torch.zeros(shape, dtype=getattr(torch, dtype)) for shape, dtype in
                 ((s.shape, s.dtype) if hasattr(s, "shape") else s for s in spec.values()))


def export_onnx(name: str, out: Path) -> dict:
    import onnxruntime as ort
    import torch
    from qai_hub_models.utils.onnx.helpers import safe_torch_onnx_export

    model = _model(name)
    io = {}
    for comp_name, comp in (("encoder", model.encoder), ("decoder", model.decoder)):
        ins, outs = comp.get_input_spec(), list(comp.get_output_spec())
        example = _zeros(ins)
        path = out / f"{comp_name}.onnx"
        t0 = time.time()
        with torch.no_grad():
            safe_torch_onnx_export(comp, example, str(path), input_names=list(ins), output_names=outs,
                                   opset_version=17)
        # sanity: ONNX (CPU) vs torch on random inputs of the same spec
        rng = np.random.default_rng(0)
        feeds = {}
        for (k, v), ex in zip(ins.items(), example):
            arr = ex.numpy()
            feeds[k] = arr if arr.dtype != np.float32 or k == "attention_mask" else \
                rng.standard_normal(arr.shape).astype(np.float32) * 0.5
        with torch.no_grad():
            ref = comp(*[torch.from_numpy(v) for v in feeds.values()])
        flat = []
        def _flatten(x):
            if isinstance(x, (tuple, list)):
                for y in x:
                    _flatten(y)
            else:
                flat.append(x.numpy())
        _flatten(ref)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, feeds)
        diff = max(float(np.max(np.abs(a - b))) for a, b in zip(got, flat))
        print(f"{comp_name}: {path.relative_to(ROOT)}  {path.stat().st_size / 1e6:.1f} MB  "
              f"export {time.time() - t0:.1f} s  onnx-vs-torch max|diff| {diff:.2e}")
        io[comp_name] = {"file": path.name,
                         "inputs": {k: [list(v.shape), v.dtype] if hasattr(v, "shape") else [list(v[0]), v[1]]
                                    for k, v in ins.items()},
                         "outputs": outs}
    return {"io": io, "config": model.config, "hf_id": model.hf_source}


def export_assets(hf_id: str, config, io: dict, out: Path) -> None:
    from qai_hub_models.models.templates.hf_whisper import model as tmpl
    from transformers import GenerationConfig, WhisperFeatureExtractor, WhisperTokenizer
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    fe = WhisperFeatureExtractor.from_pretrained(hf_id)
    np.save(out / "mel_filters.npy", fe.mel_filters.astype(np.float32))       # (n_fft//2+1, n_mels)

    tok = WhisperTokenizer.from_pretrained(hf_id)
    byte_decoder = {c: b for b, c in bytes_to_unicode().items()}
    first_special = tok.convert_tokens_to_ids("<|endoftext|>")
    table = []
    for i in range(first_special):
        piece = tok.convert_ids_to_tokens(i)
        table.append(bytes(byte_decoder[c] for c in piece).hex())
    ids = lambda *names: {n: tok.convert_tokens_to_ids(n) for n in names}  # noqa: E731
    special = ids("<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|translate|>",
                  "<|notimestamps|>", "<|nospeech|>", "<|startofprev|>", "<|0.00|>")
    (out / "tokens.json").write_text(json.dumps({"first_special_id": first_special, "special": special,
                                                 "vocab_size": len(tok), "bytes_hex": table}), encoding="utf-8")

    gen = GenerationConfig.from_pretrained(hf_id)
    meta = {
        "hf_id": hf_id, "source": "qai_hub_models.models.templates.hf_whisper",
        "sample_rate": fe.sampling_rate, "n_fft": fe.n_fft, "hop_length": fe.hop_length,
        "n_mels": fe.feature_size, "n_samples": fe.n_samples, "n_frames": fe.nb_max_frames,
        "decode_len": tmpl.MEAN_DECODE_LEN, "audio_emb_len": tmpl.AUDIO_EMB_LEN, "mask_neg": tmpl.MASK_NEG,
        "decoder_layers": config.decoder_layers, "decoder_heads": config.decoder_attention_heads,
        "d_model": config.d_model, "vocab_size": config.vocab_size,
        "special": special, "suppress_tokens": list(gen.suppress_tokens or []),
        "begin_suppress_tokens": list(gen.begin_suppress_tokens or []), "io": io,
    }
    (out / "whisper.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"assets: mel_filters.npy {fe.mel_filters.shape}, tokens.json ({len(table)} byte tokens), whisper.json")


def submit_aihub(name: str, device: str) -> None:
    from bench.aihub_profile import JOBS_FILE, _record, append_records
    export = importlib.import_module(f"qai_hub_models.models.{name}.export")
    runtime = "precompiled_qnn_onnx"
    with tempfile.TemporaryDirectory() as tmp:
        args = export.build_parser().parse_args([
            "--device", device, "--runtime", runtime, "--skip-inferencing", "--skip-downloading",
            "--skip-summary", "--output-dir", tmp])
        res = export.export_model(**vars(args))
    records = []
    for kind in ("compile", "link", "profile"):
        for comp, job in (getattr(res, f"{kind}_jobs", None) or {}).items():
            records.append(_record(kind, job, f"{name}_{comp}", device, runtime, source="qai_hub_models export",
                                   wants=[]))
            print(f"{kind:<8} {name}_{comp:<8} {job.job_id}  {job.url}")
    append_records(JOBS_FILE, records)
    print(f"recorded {len(records)} job(s) in {JOBS_FILE.relative_to(ROOT)}")


def fetch_faster_whisper(name: str) -> None:
    """Download the CTranslate2 model for the dev-only faster_whisper backend (runtime never downloads)."""
    from huggingface_hub import snapshot_download
    out = ROOT / "weights" / "asr" / "faster_whisper" / name
    snapshot_download(f"Systran/faster-whisper-{name}", local_dir=str(out))
    print(f"faster-whisper {name}: {out.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="whisper_base", help="qai_hub_models model id (whisper_tiny, whisper_base, ...)")
    p.add_argument("--list", action="store_true", help="list the Whisper models in the installed qai_hub_models")
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--aihub", action="store_true", help="also compile + profile on AI Hub (bench.aihub_device)")
    p.add_argument("--faster-whisper", metavar="MODEL", help="only download faster-whisper MODEL (tiny.en, base.en)")
    args = p.parse_args(argv)
    if args.faster_whisper:
        fetch_faster_whisper(args.faster_whisper)
        return 0
    if args.list:
        import qai_hub_models
        base = Path(qai_hub_models.__file__).parent / "models"
        print("\n".join(sorted(d.name for d in base.iterdir() if d.is_dir() and "whisper" in d.name)))
        return 0
    out = ROOT / "weights" / "asr" / args.model
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_onnx:
        r = export_onnx(args.model, out)
        export_assets(r["hf_id"], r["config"], r["io"], out)
    if args.aihub:
        from kavach_config import get_config
        submit_aihub(args.model, get_config().bench.aihub_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
