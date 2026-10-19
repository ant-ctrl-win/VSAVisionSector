#!/usr/bin/env python3
"""Export the pinned MobileNetV4 Small backbone encoder to LiteRT and check parity.

The encoder output is the spatial global average pool of the backbone feature
map (`[N, 960]`), not the pre-logits hidden features.

Usage (Linux/Colab):
    python tools/mobilenetv4_export/export_encoder.py --samples 4 --seed 0

The script does not perform image preprocessing. It generates deterministic
float32 NHWC tensors, runs the same values through PyTorch (as NCHW) and the
exported LiteRT model (as NHWC), and writes a .tflite plus a JSON manifest.
"""

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import litert_torch
from ai_edge_litert.interpreter import Interpreter

MODEL_ID = "timm/mobilenetv4_conv_small.e2400_r224_in1k"
HF_REVISION = "331fb803779522b685cf942e15f914fb6741c1eb"
EXPECTED_EMBEDDING_DIM = 960
INPUT_SIZE = 224
EXPECTED_INPUT_SHAPE = (1, INPUT_SIZE, INPUT_SIZE, 3)
EXPECTED_OUTPUT_SHAPE = (1, EXPECTED_EMBEDDING_DIM)


class PooledEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.forward_features(x)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)


def build_encoder():
    print(f"Loading {MODEL_ID}@{HF_REVISION}", flush=True)
    model = timm.create_model(
        f"hf-hub:{MODEL_ID}@{HF_REVISION}",
        pretrained=True,
        num_classes=0,
    )
    model.eval()

    if int(model.num_features) != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"unexpected num_features: {model.num_features}, "
            f"expected {EXPECTED_EMBEDDING_DIM}"
        )

    return PooledEncoder(model).eval()


def make_deterministic_inputs(samples, seed):
    rng = np.random.default_rng(seed)
    nhwc = rng.standard_normal(
        (samples, INPUT_SIZE, INPUT_SIZE, 3)
    ).astype(np.float32) * np.float32(2.0)
    nhwc = np.ascontiguousarray(nhwc, dtype=np.float32)
    nchw = np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2), dtype=np.float32)
    return nhwc, nchw


def pytorch_forward(encoder, nchw):
    with torch.inference_mode():
        output = encoder(torch.from_numpy(nchw))
    return output.detach().cpu().numpy().astype(np.float32, copy=True)


def verify_backbone_contract(encoder, nchw_sample):
    with torch.inference_mode():
        features = encoder.model.forward_features(torch.from_numpy(nchw_sample))

    if features.dim() != 4:
        raise RuntimeError(
            f"forward_features must return rank 4, got shape {tuple(features.shape)}"
        )
    if int(features.shape[1]) != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"unexpected backbone channels: {features.shape[1]}, "
            f"expected {EXPECTED_EMBEDDING_DIM}"
        )

    pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
    expected_shape = (int(nchw_sample.shape[0]), EXPECTED_EMBEDDING_DIM)
    if tuple(pooled.shape) != expected_shape:
        raise RuntimeError(
            f"unexpected pooled embedding shape {tuple(pooled.shape)}, "
            f"expected {expected_shape}"
        )

    print(f"Backbone feature map shape: {tuple(features.shape)}")
    print(f"Global-average-pooled embedding shape: {tuple(pooled.shape)}")


def convert_encoder(encoder, nhwc_sample, tflite_path):
    channel_last_encoder = litert_torch.to_channel_last_io(
        encoder, args=[0]
    ).eval()
    sample_inputs = (
        torch.from_numpy(
            np.ascontiguousarray(nhwc_sample, dtype=np.float32)
        ),
    )
    with torch.no_grad():
        edge_model = litert_torch.convert(channel_last_encoder, sample_inputs)
    edge_model.export(str(tflite_path))


def tflite_contract(interpreter):
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"expected one input and one output, got {len(inputs)} and {len(outputs)}"
        )

    input_detail = inputs[0]
    output_detail = outputs[0]
    input_shape = tuple(int(v) for v in input_detail["shape"])
    output_shape = tuple(int(v) for v in output_detail["shape"])

    if input_shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            f"unexpected LiteRT input shape {input_shape}, "
            f"expected {EXPECTED_INPUT_SHAPE}"
        )
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(
            f"unexpected LiteRT output shape {output_shape}, "
            f"expected {EXPECTED_OUTPUT_SHAPE}"
        )
    if input_detail["dtype"] != np.float32:
        raise RuntimeError(f"unexpected LiteRT input dtype: {input_detail['dtype']}")
    if output_detail["dtype"] != np.float32:
        raise RuntimeError(f"unexpected LiteRT output dtype: {output_detail['dtype']}")

    return input_detail, output_detail, input_shape, output_shape


def run_tflite(tflite_path, nhwc):
    interpreter = Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail, output_detail, input_shape, output_shape = tflite_contract(
        interpreter
    )

    outputs = []
    for index in range(nhwc.shape[0]):
        interpreter.set_tensor(
            input_detail["index"],
            np.ascontiguousarray(nhwc[index:index + 1], dtype=np.float32),
        )
        interpreter.invoke()
        outputs.append(
            interpreter.get_tensor(output_detail["index"]).astype(
                np.float32, copy=True
            )
        )

    return np.concatenate(outputs, axis=0), input_shape, output_shape


def parity_metrics(reference, exported):
    if reference.shape != exported.shape:
        raise RuntimeError(
            f"parity shape mismatch: {reference.shape} vs {exported.shape}"
        )

    reference64 = reference.astype(np.float64)
    exported64 = exported.astype(np.float64)
    absolute_error = np.abs(reference64 - exported64)

    reference_norm = np.linalg.norm(reference64, axis=1)
    exported_norm = np.linalg.norm(exported64, axis=1)
    denominator = reference_norm * exported_norm
    cosine = np.divide(
        np.sum(reference64 * exported64, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    metrics = {
        "samples": int(reference.shape[0]),
        "embedding_dim": int(reference.shape[1]),
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "cosine_similarity_mean": float(cosine.mean()),
        "cosine_similarity_min": float(cosine.min()),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite parity metrics: {metrics}")
    return metrics


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions():
    names = (
        "torch",
        "torchvision",
        "timm",
        "litert-torch",
        "ai-edge-litert",
        "numpy",
        "huggingface_hub",
        "safetensors",
    )
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(
        description="Export the pinned MobileNetV4 Small pooled encoder to LiteRT and report parity."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out",
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_encoder()
    nhwc, nchw = make_deterministic_inputs(args.samples, args.seed)
    verify_backbone_contract(encoder, nchw)

    single_output = pytorch_forward(encoder, nchw[:1])
    print(f"PyTorch single-sample output shape: {tuple(single_output.shape)}")
    if tuple(single_output.shape) != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(
            f"unexpected PyTorch output shape {single_output.shape}, "
            f"expected {EXPECTED_OUTPUT_SHAPE}"
        )

    reference = pytorch_forward(encoder, nchw)
    expected_reference_shape = (args.samples, EXPECTED_EMBEDDING_DIM)
    if tuple(reference.shape) != expected_reference_shape:
        raise RuntimeError(
            f"unexpected PyTorch batch output shape {reference.shape}, "
            f"expected {expected_reference_shape}"
        )

    tflite_path = args.output_dir / "mobilenetv4_conv_small_encoder.tflite"
    manifest_path = args.output_dir / "mobilenetv4_conv_small_encoder.manifest.json"

    print(f"Converting encoder to {tflite_path}", flush=True)
    convert_encoder(encoder, nhwc[:1], tflite_path)

    exported, input_shape, output_shape = run_tflite(tflite_path, nhwc)
    print(f"LiteRT input shape: {input_shape}")
    print(f"LiteRT output shape: {output_shape}")

    if tuple(exported.shape) != expected_reference_shape:
        raise RuntimeError(
            f"unexpected LiteRT batch output shape {exported.shape}, "
            f"expected {expected_reference_shape}"
        )

    metrics = parity_metrics(reference, exported)
    manifest = {
        "schemaVersion": 1,
        "source": {
            "model": MODEL_ID,
            "hfRevision": HF_REVISION,
            "pretrained": True,
            "classifier": "bypassed; ImageNet classifier and pre-logits hidden head are not used",
            "embeddingDim": EXPECTED_EMBEDDING_DIM,
        },
        "input": {
            "shape": list(input_shape),
            "layout": "NHWC",
            "dtype": "float32",
            "note": "deterministic tensors only; no image resize, crop or preprocessing",
        },
        "output": {
            "shape": list(output_shape),
            "layout": "NC",
            "dtype": "float32",
            "semantic": "global average pooled backbone embedding; ImageNet logits and pre-logits head excluded",
            "pooling": "adaptive_avg_pool2d(features, 1).flatten(1)",
        },
        "parity": metrics,
        "deterministicInput": {
            "seed": args.seed,
            "samples": args.samples,
            "distribution": "numpy default_rng standard_normal * 2.0",
            "shape": list(nhwc.shape),
        },
        "tflite": {
            "file": tflite_path.name,
            "bytes": tflite_path.stat().st_size,
            "sha256": sha256_file(tflite_path),
        },
        "packages": package_versions(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "tflite": str(tflite_path),
                "manifest": str(manifest_path),
                "inputShape": list(input_shape),
                "outputShape": list(output_shape),
                "embeddingDim": EXPECTED_EMBEDDING_DIM,
                "parity": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
