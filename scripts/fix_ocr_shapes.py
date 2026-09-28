"""Write static-shape copies of the OCR models for Qualcomm AI Hub / the native backend.

    weights/ocr/det_static.onnx   input x: [1, 3, H, W]            from ocr.det_input_hw
    weights/ocr/rec_static.onnx   input x: [rec_batch, 3, 48, W]   from ocr.rec_input_hw, ocr.rec_batch

Uses onnxruntime.tools.onnx_model_utils (make_input_shape_fixed + fix_output_shapes) and the onnx
API. onnx is pinned to 1.18.0 because newer wheels are blocked by Smart App Control; don't upgrade.
Each output file is checked with onnx.checker and run once on CPU through models.runtime.

Usage:  python scripts\\fix_ocr_shapes.py   (run scripts\\fetch_ocr_models.py first)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kavach_config import get_config  # noqa: E402
from models import runtime  # noqa: E402

OCR_DIR = ROOT / "weights" / "ocr"


def _dims(value_info: onnx.ValueInfoProto) -> list:
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in value_info.type.tensor_type.shape.dim]


def fix_model(src: Path, dst: Path, shape: list[int]) -> onnx.ModelProto:
    """Fix the single graph input to `shape`, propagate output shapes, check and save."""
    model = onnx.load(str(src))
    inputs = [i for i in model.graph.input if i.name not in {init.name for init in model.graph.initializer}]
    if len(inputs) != 1:
        raise SystemExit(f"{src.name}: expected one input, found {[i.name for i in inputs]}")
    make_input_shape_fixed(model.graph, inputs[0].name, shape)
    fix_output_shapes(model)  # symbolic shape inference so outputs get concrete dims too
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(dst))
    return model


def pin_output_shapes(model: onnx.ModelProto, dst: Path, observed: list[tuple[int, ...]]) -> bool:
    """Where shape inference left an output dim symbolic, set it from an actual run. Returns True if changed."""
    changed = False
    for vi, shape in zip(model.graph.output, observed):
        dims = vi.type.tensor_type.shape.dim
        if len(dims) != len(shape):
            raise SystemExit(f"{dst.name}: output {vi.name} rank {len(dims)} != observed {shape}")
        for d, n in zip(dims, shape):
            if not d.HasField("dim_value"):
                d.dim_value = int(n)  # clears dim_param (oneof)
                changed = True
    if changed:
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, str(dst))
    return changed


def cpu_run(path: Path, name: str) -> tuple[list, float]:
    sess = runtime.create_session(path, name, provider="cpu")
    feeds = {s["name"]: np.random.default_rng(0).random(s["shape"], dtype=np.float32) for s in sess.input_specs()}
    sess.run(feeds)  # first run includes graph setup
    t0 = time.perf_counter()
    outs = sess.run(feeds)
    return [o.shape for o in outs], (time.perf_counter() - t0) * 1000


def main() -> int:
    cfg = get_config().ocr
    det_h, det_w = cfg.det_input_hw
    rec_h, rec_w = cfg.rec_input_hw
    batch = int(cfg.get("rec_batch", 8))
    for n, v in (("det H", det_h), ("det W", det_w), ("rec W", rec_w)):
        if v % 32:
            print(f"warning: {n}={v} is not a multiple of 32")

    jobs = [
        ("det", OCR_DIR / "det.onnx", OCR_DIR / "det_static.onnx", [1, 3, det_h, det_w]),
        ("rec", OCR_DIR / "rec.onnx", OCR_DIR / "rec_static.onnx", [batch, 3, rec_h, rec_w]),
    ]
    print(f"onnx {onnx.__version__}")
    for tag, src, dst, shape in jobs:
        if not src.is_file():
            print(f"missing {src.relative_to(ROOT)}; run scripts\\fetch_ocr_models.py first")
            return 1
        model = fix_model(src, dst, shape)
        out_shapes, _ = cpu_run(dst, f"ocr.{tag}_static.check")
        pinned = pin_output_shapes(model, dst, out_shapes)
        out_shapes, ms = cpu_run(dst, f"ocr.{tag}_static.check")  # timed run on the final saved file
        print(f"\n{dst.relative_to(ROOT)}  ({dst.stat().st_size / 1e6:.1f} MB, was {src.stat().st_size / 1e6:.1f} MB)"
              f"  opset {model.opset_import[0].version}  onnx.checker: ok"
              + ("  (output dims pinned from CPU run)" if pinned else ""))
        for vi in model.graph.input:
            print(f"  input   {vi.name:<28} {_dims(vi)}")
        for vi in model.graph.output:
            print(f"  output  {vi.name:<28} {_dims(vi)}")
        print(f"  CPU run: outputs {out_shapes}  {ms:.0f} ms")
    runtime.clear_registry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
