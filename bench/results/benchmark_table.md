# Kavach benchmark table

Generated 2026-09-29 00:09 +0530 by `python -m bench.local_bench` (50 timed runs after 5 warm-up runs per model; synthetic inputs only).

| | Device | Runtime |
|---|---|---|
| Snapdragon NPU (AI Hub) | Snapdragon X Elite CRD (Hexagon NPU), AI Hub hosted | ONNX Runtime 1.27.1, QAIRT v2.50.0.260828221209, Windows Windows 11 (26200), Build ID APSS.WP_HA.1.0-09400-SC8380XRELSFNWZA-8; target `precompiled_qnn_onnx` |
| Dev CPU | AMD Ryzen 7 7735HS with Radeon Graphics | onnxruntime 1.24.4 (onnxruntime-directml==1.24.4), CPUExecutionProvider |
| RTX 4050 (DirectML) | NVIDIA GeForce RTX 4050 Laptop GPU, 566.26 (DirectML adapter `dml.device_id: 1`) | onnxruntime 1.24.4, DmlExecutionProvider |

Local cells are **p50 / p95 ms**. The AI Hub cell is the profiler's estimated inference time (ms). Op split and peak memory come from the AI Hub profile. Radeon iGPU (DirectML device 0) numbers from earlier runs are not included.

## Per model

| Model | Backend | Snapdragon NPU (AI Hub) | Dev CPU | RTX 4050 (DirectML) | NPU / GPU / CPU ops (AI Hub) | Peak memory (AI Hub) |
|---|---|---|---|---|---|---|
| OCR det (1x3x736x1280) | native | 14.1 | 128.2 / 150.5 | 16.9 / 17.5 | 201 / 0 / 0 | 53.3 MB |
| OCR det (1x3x736x1280) | rapidocr | n/a - dynamic shapes, not compiled for the NPU (native backend is the NPU path) | 141.9 / 158.1 | 25.8 / 26.6 | - | - |
| OCR rec, batch 8x3x48x320 | native | 32.4 | 76.8 / 90.0 | 12.2 / 13.0 | 228 / 0 / 0 | 62.2 MB |
| OCR rec, batch 8x3x48x320 | rapidocr | n/a - dynamic shapes, not compiled for the NPU (native backend is the NPU path) | 85.9 / 112.4 | 19.8 / 21.2 | - | - |
| OCR rec, batch 8x3x48x640 | native | 64.2 | 294.2 / 338.6 | 27.1 / 27.6 | 228 / 0 / 0 | 97.2 MB |
| OCR rec, batch 8x3x48x640 | rapidocr | n/a - dynamic shapes, not compiled for the NPU (native backend is the NPU path) | 277.9 / 316.5 | 41.7 / 46.0 | - | - |
| OCR rec, batch 4x3x48x1280 | native | 107.0 | 279.3 / 382.7 | 27.0 / 28.0 | 228 / 0 / 0 | 97.2 MB |
| OCR rec, batch 4x3x48x1280 | rapidocr | n/a - dynamic shapes, not compiled for the NPU (native backend is the NPU path) | 295.1 / 327.2 | 45.2 / 46.1 | - | - |
| Whisper-Base encoder (1x80x3000 = 30 s) | aihub_whisper | 45.9 | 437.5 / 778.0 | 50.0 / 51.8 | 556 / 0 / 0 | 95.0 MB |
| Whisper-Base decoder, 1 step (200-slot KV cache) | aihub_whisper | 3.6 | 14.3 / 14.7 | 11.4 / 23.4 | 975 / 0 / 0 | 174.2 MB |

## End to end

| Pipeline | Backend | Snapdragon NPU (AI Hub) | Dev CPU | RTX 4050 (DirectML) | Notes |
|---|---|---|---|---|---|
| OCR full frame, synthetic 1280x720 (6 lines) | native | pending (not measured end to end on device) | 757.3 / 888.3 | 91.2 / 94.1 | det + rec + pre/post processing, measured |
| OCR full frame, synthetic 1280x720 (6 lines) | rapidocr | pending (not measured end to end on device) | 438.1 / 476.0 | 162.1 / 165.7 | det + rec + pre/post processing, measured |
| Whisper-Base per 5 s chunk = encoder + 20 decoder steps | aihub_whisper | 117.1 (computed from AI Hub-measured parts: 45.9 + 20 x 3.558) | 723.1 (computed from measured parts) | 278.0 (computed from measured parts) | p50 parts; excludes log-mel (~5 ms numpy) and host overhead; real-time factor = value / 5000 ms |

## AI Hub jobs

| Model | Profile job | Status |
|---|---|---|
| det_static | [j57e1nvrp](https://workbench.aihub.qualcomm.com/jobs/j57e1nvrp/) | SUCCESS |
| rec_static_1280 | [jpyok2v45](https://workbench.aihub.qualcomm.com/jobs/jpyok2v45/) | SUCCESS |
| rec_static_320 | [jpxl8re9p](https://workbench.aihub.qualcomm.com/jobs/jpxl8re9p/) | SUCCESS |
| rec_static_640 | [jprlmd1ep](https://workbench.aihub.qualcomm.com/jobs/jprlmd1ep/) | SUCCESS |
| whisper_base_decoder | [j5m01kdqg](https://workbench.aihub.qualcomm.com/jobs/j5m01kdqg/) | SUCCESS |
| whisper_base_encoder | [jpxl8rd9p](https://workbench.aihub.qualcomm.com/jobs/jpxl8rd9p/) | SUCCESS |
