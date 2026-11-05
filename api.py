#!/usr/bin/env python
from fastapi import FastAPI, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional, Any, List

import cipherTypeDetection.constants as constants
from cipherImplementations.cipher import OUTPUT_ALPHABET
from cipherTypeDetection import predict as cipherPredict
from cipherTypeDetection.ensembleModel import EnsembleModel
from cipherTypeDetection.rotorDifferentiationEnsemble import RotorDifferentiationEnsemble
from inference.registry import ONNX_MODEL_DIR, load_all


# init fast api
app = FastAPI()
models = {}

# allow cors
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  # todo: remove later


@app.on_event("startup")
async def startup_event():
    """The ONNX models are loaded from the registry with hardcoded names.
    Missing model artifacts fail at startup with a clear error. The ONNX
    artifacts are deployed out-of-band into the model directory."""
    models.update(load_all(ONNX_MODEL_DIR))


class ArchitectureError(Exception):
    def __init__(self, response):
        Exception.__init__(self)
        self.response = response

def validate_architectures(architectures):
    max_architectures = len(models.keys())
    # Warn if the provided number of architectures is out of the expected bounds
    if not 0 < len(architectures) <= max_architectures:
        response = JSONResponse(
            {
                "success": False,
                "payload": None,
                "error_msg": "The number of architectures must be between 1 and %d." %
                max_architectures},
            status_code=status.HTTP_400_BAD_REQUEST)
        raise ArchitectureError(response)

    # Warn about duplicate architectures
    if len(set(architectures)) != len(architectures):
        response = JSONResponse({"success": False,
                              "payload": None,
                              "error_msg": "Multiple architectures of the same type are not "
                              "allowed!"},
                             status_code=status.HTTP_400_BAD_REQUEST)
        raise ArchitectureError(response)

    # Check if the provided architectures are known
    for architecture in architectures:
        if architecture not in models.keys():
            response = JSONResponse(
                {
                    "success": False,
                    "payload": None,
                    "error_msg": "The architecture '%s' does not exist!" %
                    architecture},
                status_code=status.HTTP_400_BAD_REQUEST)
            raise ArchitectureError(response)

class APIResponse(BaseModel):
    """Define api response model."""
    success: bool = True
    payload: Optional[Any] = {}
    error_msg: Optional[str] = None


# The alphabet the models operate on (see cipherImplementations.cipher).
ALLOWED_CIPHERTEXT_CHARS = set(OUTPUT_ALPHABET.decode("latin-1"))


def normalize_and_validate_ciphertext(ciphertext):
    """Normalize the ciphertext for the models.

    Ciphertexts are commonly written in uppercase (ACA convention), while the
    models operate on the lowercase output alphabet, so the case is
    normalized. Characters that cannot be mapped into the output alphabet
    cannot be processed by the feature engineering, so they are rejected with
    a human-readable error.

    Parameters
    ----------
    ciphertext : str
        The raw ciphertext from the request.

    Returns
    -------
    str
        The normalized ciphertext, ready to be encoded and predicted.

    Raises
    ------
    ValueError
        If the ciphertext contains unsupported characters.
    """
    normalized = ciphertext.lower()
    unsupported = sorted(set(normalized) - ALLOWED_CIPHERTEXT_CHARS)
    if unsupported:
        raise ValueError(
            "The ciphertext contains unsupported characters: %s. "
            "Allowed are a-z, 0-9, '#' and space."
            % ", ".join(repr(c) for c in unsupported))
    return normalized


@app.exception_handler(Exception)
async def exception_handler(request, exc):
    """Define exception response format."""
    return JSONResponse({"success": False, "payload": None, "error_msg": str(
        exc)}, status_code=status.HTTP_400_BAD_REQUEST)


@app.get("/get_available_architectures", response_model=APIResponse)
async def get_available_architectures():
    visible_models = [model for model in list(models.keys()) if model != "Rotor-SVM"]
    return {"success": True, "payload": visible_models}

@app.get("/evaluate/single_line/ciphertext", response_model=APIResponse)
async def evaluate_single_line_ciphertext(ciphertext: str, architecture: List[str] = Query([])):
    rotor_only_model = models["Rotor-SVM"]

    # use plural inside function
    architectures = architecture
    try:
        validate_architectures(architectures)
    except ArchitectureError as error:
        return error.response

    try:
        # Normalize the case of the ciphertext and reject input that the
        # feature engineering cannot process (would otherwise crash with a
        # 500 on e.g. uppercase-only or punctuation-containing input).
        try:
            ciphertext = normalize_and_validate_ciphertext(ciphertext)
        except ValueError as error:
            return JSONResponse(
                {"success": False,
                 "payload": None,
                 "error_msg": str(error)},
                status_code=status.HTTP_400_BAD_REQUEST)

        # only use architectures that can predict aca ciphers
        architectures = [architecture
            for architecture in architectures
            if architecture in ("Transformer", "FFNN", "LSTM", "RF", "NB")]

        aca_cipher_types = list(range(56))
        rotor_cipher_types = list(range(56, 61))

        if len(architectures) == 0:
            return {}
        elif len(architectures) == 1:
            architecture = architectures[0]
            model = models[architecture]
        else:
            model_files = [models[architecture] for architecture in architectures]
            architecture = "Ensemble"
            model = EnsembleModel(
                model_files,
                "weighted",
                aca_cipher_types + rotor_cipher_types)

        # Embed all models in RotorDifferentiationEnsemble to improve recognition of rotor ciphers
        model = RotorDifferentiationEnsemble(architecture, model, rotor_only_model)
        architecture = "Ensemble"

        all_cipher_types = aca_cipher_types + rotor_cipher_types
        cipher_names = [constants.CIPHER_TYPES[cipher_index] for cipher_index in all_cipher_types]

        prediction = cipherPredict.predict_ciphertext_line(
            ciphertext.encode(), model, cipher_names, batch_size=128)
        # Convert to native floats so that the response is JSON serializable.
        prediction = {cipher_type: float(value) for cipher_type, value in prediction.items()}

        return {"success": True, "payload": prediction}
    except BaseException as e:
        # only use these lines for debugging. Never in production environment due
        # to security reasons!
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "payload": None,
                            "error_msg": repr(e)}, status_code=500)
