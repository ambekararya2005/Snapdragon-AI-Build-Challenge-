"""Offline tests for bench.aihub_profile (no AI Hub calls, no token needed)."""
import json
from pathlib import Path

import numpy as np
import pytest

from bench import aihub_profile as A

ROOT = Path(__file__).resolve().parent.parent
STATIC = [ROOT / "weights" / "ocr" / n for n in
          ("det_static.onnx", "rec_static_320.onnx", "rec_static_640.onnx", "rec_static_1280.onnx", "keys.txt")]
needs_static = pytest.mark.skipif(not all(p.is_file() for p in STATIC), reason="static OCR models missing")


def test_append_records_never_rewrites(tmp_path):
    f = tmp_path / "jobs.json"
    A.append_records(f, [{"job_id": "a"}])
    A.append_records(f, [{"job_id": "b"}, {"job_id": "c"}])
    assert [r["job_id"] for r in json.loads(f.read_text())] == ["a", "b", "c"]
    f.write_text('{"not": "a list"}')
    with pytest.raises(SystemExit):
        A.append_records(f, [{"job_id": "d"}])


def test_is_snapdragon_x():
    assert A.is_snapdragon_x("Snapdragon X Elite CRD", [])
    assert A.is_snapdragon_x("Some Laptop", ["chipset:qualcomm-snapdragon-x-plus-8-core"])
    assert not A.is_snapdragon_x("Samsung Galaxy S24", ["chipset:qualcomm-snapdragon-8gen3"])


def test_summarize_profile():
    profile = {
        "execution_summary": {"estimated_inference_time": 12345, "estimated_inference_peak_memory": 50 * 1024 * 1024},
        "execution_detail": [{"compute_unit": "NPU"}] * 7 + [{"compute_unit": "CPU"}] * 2,
    }
    s = A.summarize_profile(profile)
    assert s == {"inference_ms": 12.345, "peak_memory_mb": 50.0, "ops": {"NPU": 7, "GPU": 0, "CPU": 2}}
    assert A.summarize_profile({})["inference_ms"] is None


@needs_static
def test_input_specs_are_static():
    assert A.onnx_input_specs(ROOT / "weights/ocr/det_static.onnx") == {"x": ((1, 3, 736, 1280), "float32")}
    assert A.onnx_input_specs(ROOT / "weights/ocr/rec_static_1280.onnx") == {"x": ((4, 3, 48, 1280), "float32")}
    with pytest.raises(SystemExit):
        A.onnx_input_specs(ROOT / "weights/ocr/rec.onnx")        # dynamic model is refused


@needs_static
def test_synthetic_inputs_and_self_compare():
    det_in = A.synthetic_inputs("det_static")
    assert det_in["x"].shape == (1, 3, 736, 1280)
    rec_in = A.synthetic_inputs("rec_static_640")
    assert rec_in["x"].shape == (8, 3, 48, 640) and np.all(rec_in["x"][1:] == 0)

    local = A.local_cpu_outputs("rec_static_640", rec_in)
    cmp = A.compare_outputs("rec_static_640", {"out": [local[0]]}, local)   # "remote" = local: identical
    assert cmp["max_abs_diff"] == [0.0] and cmp["text_equal"] is True
    assert "482913" in cmp["local_text"]
