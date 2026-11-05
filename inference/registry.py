"""Registry of the ONNX models served by the API.

The ONNX artifacts are deployed out-of-band (e.g. by scp) into
`ONNX_MODEL_DIR`. They are exported with `util/export_models_to_onnx.py`.
"""

import os

from inference.onnx_model import OnnxModel

# The directory the ONNX model artifacts are deployed to.
ONNX_MODEL_DIR = "data/models"

# Maps the model names used by the API to the (filename, input kind) of the
# corresponding ONNX artifact.
MODEL_REGISTRY = {
    "Transformer": ("transformer_var_10000000.onnx", OnnxModel.TOKENS),
    "FFNN": ("ffnn_var_10000000.onnx", OnnxModel.STATISTICS),
    "LSTM": ("lstm_var_10000000.onnx", OnnxModel.TOKENS),
    "RF": ("rf_var_1000000.onnx", OnnxModel.STATISTICS),
    "NB": ("nb_var_10000000.onnx", OnnxModel.STATISTICS),
    "Rotor-SVM": ("svm_rotor_only_1000_16000.onnx", OnnxModel.ROTOR_STATISTICS),
}


def load_model(name, model_dir=ONNX_MODEL_DIR):
    """Load a single ONNX model by its registry name.

    If a zstd-compressed variant `<filename>.zst` exists next to the plain
    ONNX file, it is preferred, so models can be stored compressed at rest.

    Raises
    ------
    ValueError
        If `name` is not part of the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Known models: {sorted(MODEL_REGISTRY)}")
    filename, input_kind = MODEL_REGISTRY[name]
    path = os.path.join(model_dir, filename)
    if os.path.exists(path + ".zst"):
        path = path + ".zst"
    return OnnxModel(path, name, input_kind)


def load_all(models_dir=ONNX_MODEL_DIR):
    """Load all registered ONNX models.

    Raises
    ------
    FileNotFoundError
        If one of the model artifacts is missing, so that the API fails at
        startup with a clear error instead of failing per request.
    """
    models = {}
    for name in MODEL_REGISTRY:
        filename, _ = MODEL_REGISTRY[name]
        path = os.path.join(models_dir, filename)
        if not (os.path.exists(path) or os.path.exists(path + ".zst")):
            raise FileNotFoundError(
                f"ONNX model not found: {path}[.zst]. "
                f"Deploy the model artifacts into {models_dir} "
                f"(export them with util/export_models_to_onnx.py).")
        models[name] = load_model(name, models_dir)
    return models
