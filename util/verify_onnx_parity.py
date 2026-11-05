#!/usr/bin/env python3
"""Verify that the ONNX models serve identical predictions to the original
models they were exported from.

For every model of the API registry (`inference/registry.py`) the checker feeds
the same input the serving path uses into

- the original model (Keras `.h5` file or scikit-learn pickle), and
- the exported ONNX model,

and compares the resulting probability vectors.

The checker needs the original frameworks (TensorFlow, scikit-learn) and the
original model files, so run it in the training environment:

    python3 util/verify_onnx_parity.py
    python3 util/verify_onnx_parity.py --model-dir data/models --originals-dir /srv/ncid/models --tolerance 1e-3
    python3 util/verify_onnx_parity.py --only FFNN NB --ciphertext "abc..."

The exit code is 0 if every model is within the tolerance, 1 otherwise.
"""

import argparse
import os
import pickle
import sys

import numpy as np

from cipherImplementations.cipher import OUTPUT_ALPHABET, UNKNOWN_SYMBOL_NUMBER
from cipherTypeDetection import featureCalculations
from cipherTypeDetection.config import Backend
from cipherTypeDetection.models import predictionHelper
from cipherTypeDetection.preprocessing import pad_line
from inference.registry import MODEL_REGISTRY, load_model as load_onnx_model
from util.textutils import map_text_into_numberspace

DEFAULT_CIPHERTEXT = (
    "yingraobhoarhthosortmkicwhaslcbpirpocuedcfthcezvoryyrsrdyaffcleaetiaaeuhtyegeadsneanmatedbtrdndres"
)

# Maps the registry model names to the original (non-ONNX) model files.
SOURCE_FILES = {
    "Transformer": "transformer_var_10000000.h5",
    "FFNN": "ffnn_var_10000000.h5",
    "LSTM": "lstm_var_10000000.h5",
    "RF": "rf_var_1000000.h5",
    "NB": "nb_var_10000000.h5",
    "Rotor-SVM": "svm_rotor_only_1000_16000.h5",
}


def build_model_inputs(ciphertext):
    """Compute the model inputs of the serving path for one line of ciphertext.

    Returns
    -------
    (numberspace, statistics, rotor_statistics) : tuple
        The numberspace sequence, the 724-dim feature engineering statistics
        and the 144-dim rotor statistics of the padded line.
    """
    if isinstance(ciphertext, str):
        ciphertext = ciphertext.encode()
    line = pad_line(ciphertext)
    numberspace = map_text_into_numberspace(line, OUTPUT_ALPHABET, UNKNOWN_SYMBOL_NUMBER)
    statistics = featureCalculations.calculate_statistics(numberspace)
    rotor_statistics = featureCalculations.calculate_rotor_statistics(numberspace)
    return numberspace, statistics, rotor_statistics


def load_original(name, model_dir, originals_dir):
    """Load the original (non-ONNX) model of `name`.

    Returns
    -------
    (model, error) : tuple
        The loaded model and `None`, or `None` and an error message if the
        model file does not exist.
    """
    path = os.path.join(originals_dir, SOURCE_FILES[name])
    if not os.path.exists(path):
        return None, f"original model not found: {path}"

    if name in ("FFNN", "LSTM"):
        import tensorflow as tf
        return tf.keras.models.load_model(path), None
    if name == "Transformer":
        import tensorflow as tf
        from cipherTypeDetection.transformer import (
            MultiHeadSelfAttention,
            TransformerBlock,
            TokenAndPositionEmbedding,
        )
        return tf.keras.models.load_model(
            path,
            custom_objects={
                "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "MultiHeadSelfAttention": MultiHeadSelfAttention,
                "TransformerBlock": TransformerBlock,
            },
        ), None
    # scikit-learn models are pickled (file name ends in .h5 for historical
    # reasons)
    with open(path, "rb") as f:
        return pickle.load(f), None


def original_prediction(name, model, numberspace, statistics, rotor_statistics):
    """Run the original model exactly the way the old serving path did."""
    if name == "FFNN":
        return predictionHelper.predict_ffnn(model, [statistics], 128, Backend.KERAS)
    if name == "LSTM":
        return predictionHelper.predict_lstm(model, [numberspace], 128, Backend.KERAS)
    if name == "Transformer":
        return predictionHelper.predict_transformer(model, [numberspace], 128)
    if name in ("RF", "NB"):
        return model.predict_proba(np.asarray([statistics]))
    if name == "Rotor-SVM":
        return model.predict_proba(np.asarray([rotor_statistics]))
    raise ValueError(f"Unknown model: {name}")


def onnx_prediction(name, model_dir, numberspace, statistics, rotor_statistics):
    """Run the ONNX model via the framework-free inference package."""
    onnx_model = load_onnx_model(name, model_dir)
    if name in ("RF", "NB", "Rotor-SVM"):
        if name == "Rotor-SVM":
            return onnx_model.predict_proba(np.asarray([rotor_statistics]))
        return onnx_model.predict_proba(np.asarray([statistics]))
    return onnx_model.predict([statistics], [numberspace], 128)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify that the ONNX models match the original models.")
    parser.add_argument("--model-dir", default="data/models",
                        help="Directory with the ONNX model files.")
    parser.add_argument("--originals-dir", default=None,
                        help="Directory with the original (non-ONNX) model "
                             "files. Defaults to --model-dir.")
    parser.add_argument("--ciphertext", default=DEFAULT_CIPHERTEXT,
                        help="The line of ciphertext to compare on.")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Maximum allowed absolute difference of the "
                             "probability vectors (default: 1e-3).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only check the given model names "
                             f"(default: all of {sorted(MODEL_REGISTRY)}).")
    return parser.parse_args()


def main():
    args = parse_args()
    names = args.only if args.only else list(MODEL_REGISTRY)
    for name in names:
        if name not in MODEL_REGISTRY:
            print(f"Unknown model name: {name} (known: {sorted(MODEL_REGISTRY)})")
            sys.exit(2)

    numberspace, statistics, rotor_statistics = build_model_inputs(args.ciphertext)
    originals_dir = args.originals_dir if args.originals_dir else args.model_dir

    print(f"ONNX parity check (tolerance={args.tolerance:g}, "
          f"model_dir={args.model_dir})")
    any_failure = False
    for name in names:
        status = "OK"
        detail = None
        max_abs_diff = None
        try:
            original, error = load_original(name, args.model_dir, originals_dir)
            if error:
                status, detail = "SKIP", error
            else:
                pred_original = original_prediction(
                    name, original, numberspace, statistics, rotor_statistics)
                pred_onnx = onnx_prediction(
                    name, args.model_dir, numberspace, statistics, rotor_statistics)
                original = np.asarray(pred_original, dtype=np.float64).ravel()
                onnx = np.asarray(pred_onnx, dtype=np.float64).ravel()
                if original.shape != onnx.shape:
                    status = "FAIL"
                    detail = f"shape mismatch: original {original.shape} vs onnx {onnx.shape}"
                else:
                    max_abs_diff = float(np.max(np.abs(original - onnx)))
                    if max_abs_diff > args.tolerance:
                        status, detail = "FAIL", "difference exceeds tolerance"
        except Exception as e:  # noqa: BLE001
            status, detail = "ERROR", repr(e)

        if status in ("FAIL", "ERROR"):
            any_failure = True
        diff_str = f"max_abs_diff={max_abs_diff:.3e}" if max_abs_diff is not None else (detail or "")
        print(f"  {name:<15} {status:<6} {diff_str}")

    print("Result:", "FAIL" if any_failure else "PASS")
    sys.exit(1 if any_failure else 0)


if __name__ == "__main__":
    main()
