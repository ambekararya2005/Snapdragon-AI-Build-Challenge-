"""Copy the OCR ONNX models that ship inside rapidocr_onnxruntime into weights/ocr/ (no network).

    weights/ocr/det.onnx   <- ch_PP-OCRv4_det_infer.onnx          (text detection, DB)
    weights/ocr/rec.onnx   <- ch_PP-OCRv4_rec_infer.onnx          (text recognition, CTC)
    weights/ocr/cls.onnx   <- ch_ppocr_mobile_v2.0_cls_infer.onnx (0/180 deg classifier)
    weights/ocr/keys.txt   <- character dict, extracted from rec.onnx metadata ("character");
                              rapidocr 1.4.x ships no separate keys file

These exact files are what we compile/profile on Qualcomm AI Hub and use in the native backend.

Usage:  python scripts\\fetch_ocr_models.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import runtime  # noqa: E402  (after sys.path setup)

OUT_DIR = ROOT / "weights" / "ocr"
FILES = {
    "det.onnx": "ch_PP-OCRv4_det_infer.onnx",
    "rec.onnx": "ch_PP-OCRv4_rec_infer.onnx",
    "cls.onnx": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
}


def package_models_dir() -> Path:
    spec = importlib.util.find_spec("rapidocr_onnxruntime")
    if spec is None or not spec.origin:
        raise SystemExit("rapidocr_onnxruntime is not installed (pip install -r requirements.txt)")
    return Path(spec.origin).parent / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(force: bool = False) -> dict[str, Path]:
    src_dir = package_models_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for dst_name, src_name in FILES.items():
        src, dst = src_dir / src_name, OUT_DIR / dst_name
        if not src.is_file():
            raise SystemExit(f"missing in package: {src}")
        if force or not dst.is_file() or sha256(src) != sha256(dst):
            shutil.copy2(src, dst)
            print(f"copied  {src_name} -> {dst.relative_to(ROOT)}")
        else:
            print(f"ok      {dst.relative_to(ROOT)} (already up to date)")
        out[dst_name] = dst

    # Character dict lives in the rec model's metadata.
    rec = runtime.create_session(out["rec.onnx"], "ocr.rec.inspect", provider="cpu")
    meta = rec.session.get_modelmeta().custom_metadata_map
    if "character" not in meta:
        raise SystemExit("rec.onnx has no 'character' metadata; cannot build keys.txt")
    keys = OUT_DIR / "keys.txt"
    keys.write_text(meta["character"], encoding="utf-8")
    n_chars = len(meta["character"].splitlines())
    print(f"wrote   {keys.relative_to(ROOT)} ({n_chars} characters from rec.onnx metadata)")
    out["keys.txt"] = keys
    return out


def describe(paths: dict[str, Path]) -> None:
    print()
    for name in FILES:
        path = paths[name]
        sess = runtime.create_session(path, f"ocr.{path.stem}.inspect", provider="cpu")
        size_mb = path.stat().st_size / 1e6
        print(f"{path.relative_to(ROOT)}  ({size_mb:.1f} MB, sha256 {sha256(path)[:12]})")
        for spec in sess.input_specs():
            print(f"  input   {spec['name']:<10} {spec['shape']}  {spec['dtype']}")
        for spec in sess.output_specs():
            print(f"  output  {spec['name']:<10} {spec['shape']}  {spec['dtype']}")
    runtime.clear_registry()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Copy rapidocr's bundled ONNX models into weights/ocr/")
    p.add_argument("--force", action="store_true", help="overwrite even if identical")
    describe(fetch(p.parse_args().force))
