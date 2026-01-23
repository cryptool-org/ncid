#!/usr/bin/env python
import os
import tensorflow as tf
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

from fastapi import FastAPI, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional, Any, List
from types import SimpleNamespace

import cipherTypeDetection.eval as cipherEval
import cipherTypeDetection.config as config
from cipherTypeDetection.rotorDifferentiationEnsemble import RotorDifferentiationEnsemble
from cipherTypeDetection.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding
from cipherTypeDetection.ensembleModel import EnsembleModel
from cipherTypeDetection.models.modelFile import ModelFile

import pandas as pd


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
    """The models are loaded with hardcoded names. Change in future if multiple models are available."""
    model_path = "data/models"
    models["Transformer"] = (
        tf.keras.models.load_model(
            os.path.join(model_path, "transformer_var_10000000.h5"),
            custom_objects={
                'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                'TransformerBlock': TransformerBlock}),
        False,
        True)
    models["FFNN"] = (
        tf.keras.models.load_model(
            os.path.join(model_path, "ffnn_var_10000000.h5")
            ),
        True,
        False)
    models["LSTM"] = (
        tf.keras.models.load_model(
            os.path.join(model_path, "lstm_var_10000000.h5")),
        False,
        True)
    optimizer = Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        epsilon=config.epsilon,
        amsgrad=config.amsgrad)
    for _, item in models.items():
        item[0].compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
    with open(os.path.join(model_path, "rf_var_1000000.h5"), "rb") as f:
        models["RF"] = (pickle.load(f), True, False)
    with open(os.path.join(model_path, "nb_var_10000000.h5"), "rb") as f:
        models["NB"] = (pickle.load(f), True, False)
    with open(os.path.join(model_path, "svm_rotor_only_1000_16000.h5"), "rb") as f:
        models["Rotor-SVM"] = pickle.load(f)

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
            model, feature_engineering, pad_input = models[architecture]
            config.FEATURE_ENGINEERING = feature_engineering
            config.PAD_INPUT = pad_input
        else:
            model_files = []
            for architecture in architectures:
                # Still use KERAS backend for DL models for now. Has to be changed in conjunction with the model loading code!
                backend = config.Backend.KERAS
                if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
                    backend = config.Backend.SCIKIT
                file = ModelFile(models[architecture][0], architecture, backend)
                model_files.append(file)
            architecture = "Ensemble"
            model = EnsembleModel(
                model_files,
                "weighted",
                aca_cipher_types + rotor_cipher_types)
        
        # Embed all models in RotorDifferentiationEnsemble to improve recognition of rotor ciphers
        model = RotorDifferentiationEnsemble(architecture, model, rotor_only_model)
        architecture = "Ensemble"
        
        all_cipher_types = aca_cipher_types + rotor_cipher_types
        cipher_names = [config.CIPHER_TYPES[cipher_index] for cipher_index in all_cipher_types]

        eval_args = SimpleNamespace(
            ciphertext=ciphertext,
            # todo: needs fileupload first (either set ciphertext OR file, never both)
            file=None,
            ciphers=cipher_names,
            batch_size=128,
            verbose=False
        )

        # Still use KERAS backend for now
        backend = config.Backend.KERAS

        prediction = cipherEval.predict_single_line(eval_args, model, architecture, backend)
    
        return {"success": True, "payload": prediction}
    except BaseException as e:
        # only use these lines for debugging. Never in production environment due
        # to security reasons!
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "payload": None,
                            "error_msg": repr(e)}, status_code=500)

