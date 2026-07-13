#!/usr/bin/env python3
"""
export_edge.py — Export a trained detector for EDGE devices (Jetson Nano / Raspberry Pi) and
benchmark size + inference latency (P7). Supports a sklearn joblib model (URL baseline/fusion).

Exports to ONNX (via skl2onnx) when available; always measures model size and CPU latency.
On the device: convert ONNX -> TensorRT (Jetson) or run with ONNX Runtime / TFLite (Raspberry Pi).

INSTALL: pip install scikit-learn joblib onnx skl2onnx onnxruntime   (onnx parts optional)
RUN:
  python export_edge.py --model models/url_rf.joblib --dim 22 --out models/url_rf.onnx --bench 2000
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import joblib


def size_mb(path):
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


def bench_latency(predict_fn, dim, n=2000):
    X = np.random.rand(1, dim).astype(np.float32)
    predict_fn(X)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        predict_fn(X)
    dt = (time.perf_counter() - t0) / n * 1000
    return dt  # ms/inference (batch=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="joblib file from train_url_baseline/fusion")
    ap.add_argument("--dim", type=int, required=True, help="Number of input features")
    ap.add_argument("--out", default="models/model.onnx")
    ap.add_argument("--bench", type=int, default=2000)
    args = ap.parse_args()

    obj = joblib.load(args.model)
    clf = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    print(f"sklearn model size = {size_mb(args.model):.3f} MB")
    lat = bench_latency(lambda X: clf.predict(X), args.dim, args.bench)
    print(f"sklearn CPU latency = {lat:.3f} ms/inference (batch=1, {args.bench} runs)")

    # ONNX export (optional)
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_sklearn(clf, initial_types=[("input", FloatTensorType([None, args.dim]))])
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"ONNX exported -> {args.out} ({size_mb(args.out):.3f} MB)")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            lat2 = bench_latency(lambda X: sess.run(None, {name: X}), args.dim, args.bench)
            print(f"ONNXRuntime CPU latency = {lat2:.3f} ms/inference")
        except Exception as e:
            print(f"(onnxruntime bench skipped: {type(e).__name__})")
    except Exception as e:
        print(f"(ONNX export skipped: install onnx+skl2onnx — {type(e).__name__})")

    print("On device: TensorRT (Jetson, FP16/INT8) or ONNXRuntime/TFLite (Raspberry Pi); "
          "measure power via INA219/wall-meter and thermal under sustained load.")


if __name__ == "__main__":
    main()
