"""Build a tiny ONNX model for runtime/provider smoke tests.

input [1,3,224,224] float32 -> Conv -> Relu -> GlobalAveragePool -> Flatten -> Gemm -> logits [1,10]

Usage:  python scripts\\make_dummy_model.py [--out weights\\dummy.onnx]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "weights" / "dummy.onnx"
OPSET = 17
IR_VERSION = 8  # matches opset 17; keeps the model loadable by older onnxruntime builds


def build_dummy_model(out: str | Path = DEFAULT_OUT, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    conv_w = numpy_helper.from_array(rng.standard_normal((8, 3, 3, 3)).astype(np.float32) * 0.1, "conv_w")
    conv_b = numpy_helper.from_array(np.zeros(8, np.float32), "conv_b")
    fc_w = numpy_helper.from_array(rng.standard_normal((10, 8)).astype(np.float32) * 0.1, "fc_w")
    fc_b = numpy_helper.from_array(np.zeros(10, np.float32), "fc_b")

    nodes = [
        helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("GlobalAveragePool", ["relu"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "kavach_dummy",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 10])],
        initializer=[conv_w, conv_b, fc_w, fc_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="kavach")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build weights/dummy.onnx for runtime smoke tests")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    path = build_dummy_model(p.parse_args().out)
    print(f"wrote {path} ({path.stat().st_size} bytes, opset {OPSET})")
