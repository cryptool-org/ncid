#!/usr/bin/env python3
"""Export a model file to ONNX using existing project loading logic.

Examples
--------
python3 util/export_models_to_onnx.py \
  --model data/models/ffnn_var_10000000.h5 \
  --architecture FFNN \
  --backend keras

python3 util/export_models_to_onnx.py \
  --model data/models/t30.pth \
  --architecture LSTM \
  --backend pytorch \
  --input-shape 1,100
"""

import argparse
import os
from pathlib import Path

import numpy as np

from cipherTypeDetection.config import Backend
from cipherTypeDetection.models.modelFile import ModelFile


def parse_args():
    parser = argparse.ArgumentParser(description="Load and export model as ONNX.")
    parser.add_argument("--model", required=True, help="Path to input model file.")
    parser.add_argument(
        "--architecture",
        required=True,
        help="Model architecture, e.g. FFNN, LSTM, Transformer, RF, NB.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "keras", "pytorch", "scikit"),
        default="auto",
        help="Model backend. 'auto' follows eval.py logic.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX file path. Defaults to <model_basename>.onnx.",
    )
    parser.add_argument(
        "--input-shape",
        default=None,
        help="Comma-separated input shape, e.g. 1,100 or 1,128,1.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def parse_shape(input_shape_arg):
    if not input_shape_arg:
        return None
    try:
        shape = tuple(int(x.strip()) for x in input_shape_arg.split(","))
    except ValueError as exc:
        raise ValueError("--input-shape must be comma-separated integers.") from exc
    if len(shape) == 0 or any(v <= 0 for v in shape):
        raise ValueError("--input-shape must contain positive integers.")
    return shape


def infer_backend(model_path, architecture, backend_arg):
    if backend_arg != "auto":
        mapping = {
            "keras": Backend.KERAS,
            "pytorch": Backend.PYTORCH,
            "scikit": Backend.SCIKIT,
        }
        return mapping[backend_arg]

    if model_path.endswith(".pth"):
        return Backend.PYTORCH
    if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN", "SVM-Rotor"):
        return Backend.SCIKIT
    return Backend.KERAS


def output_path_for(model_path, output_arg):
    if output_arg:
        return output_arg
    stem = Path(model_path).stem
    return str(Path(model_path).with_name(f"{stem}.onnx"))


def export_keras(model, output_path, opset, input_shape):
    import tensorflow as tf
    import tf2onnx

    if input_shape is None:
        # Use model input shape when available.
        model_shape = model.input_shape
        if isinstance(model_shape, list):
            model_shape = model_shape[0]
        if model_shape is None:
            raise ValueError("Unable to infer Keras input shape. Provide --input-shape.")
        if model_shape[0] is None:
            model_shape = tuple([1] + list(model_shape[1:]))
        input_shape = tuple(model_shape)

    # Default to int32 for sequence models and float32 otherwise.
    if len(input_shape) >= 2 and model.dtype in ("int32", "int64"):
        input_dtype = tf.int32
    else:
        input_dtype = tf.float32

    spec = (tf.TensorSpec(input_shape, input_dtype, name="input"),)
    _, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=output_path,
    )


def export_pytorch(model, architecture, output_path, opset, input_shape):
    import torch

    if input_shape is None:
        if architecture == "FFNN" and hasattr(model, "input_size"):
            input_shape = (1, int(model.input_size))
        else:
            raise ValueError(
                "Missing --input-shape for this PyTorch model "
                "(example for LSTM: --input-shape 1,100)."
            )

    if architecture == "LSTM":
        dummy = torch.randint(low=0, high=10, size=input_shape, dtype=torch.long)
    else:
        dummy = torch.randn(*input_shape, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset,
        )


def export_scikit(model, output_path, opset, input_shape):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    from util.strip_zero_class_weights import strip_zero_class_weights

    if input_shape is None:
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None:
            raise ValueError(
                "Unable to infer scikit input size. Provide --input-shape, e.g. 1,128."
            )
        input_shape = (1, int(n_features))

    if len(input_shape) != 2:
        raise ValueError("scikit ONNX export expects 2D shape [batch, features].")

    initial_types = [("input", FloatTensorType(list(input_shape)))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=opset)
    # Tree ensembles store one weight per (leaf, class) pair, almost all of
    # them zero for multi-class forests. Stripping them is lossless (they add
    # nothing to the tree sums) and shrinks the artifact dramatically.
    stripped = strip_zero_class_weights(onnx_model)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    if stripped:
        print(f"Stripped {stripped:,} zero-weight class entries from the "
              f"tree ensemble.")


def main():
    args = parse_args()
    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    backend = infer_backend(model_path, args.architecture, args.backend)
    output_path = output_path_for(model_path, args.output)
    input_shape = parse_shape(args.input_shape)

    model = ModelFile(model_path, args.architecture, backend).load_model()

    if backend == Backend.KERAS:
        export_keras(model, output_path, args.opset, input_shape)
    elif backend == Backend.PYTORCH:
        export_pytorch(model, args.architecture, output_path, args.opset, input_shape)
    elif backend == Backend.SCIKIT:
        export_scikit(model, output_path, args.opset, input_shape)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    print(f"Exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()
