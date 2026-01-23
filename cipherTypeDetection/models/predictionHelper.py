"""Define helper functions to perform predictions on models given statistics or ciphertext."""

import numpy as np
import tensorflow as tf
import torch

from cipherTypeDetection.config import Backend
from util.utils import get_model_input_length, get_pytorch_device
from cipherTypeDetection.cipherStatisticsDataset import pad_sequences


def split_ciphertext_to_model_input_length(model, architecture, ciphertexts):
    """Split `ciphertexts` to match the supported input length of `model`"""

    def pad_and_split_single_ciphertext(ciphertext):
        if len(ciphertext) < input_length:
            ciphertext = pad_sequences([list(ciphertext)], maxlen=input_length)[0]
        return [
            ciphertext[input_length * j : input_length * (j + 1)]
            for j in range(len(ciphertext) // input_length)
        ]

    input_length = get_model_input_length(model, architecture)
    split_ciphertexts = []
    if isinstance(ciphertexts, list):
        for ciphertext in ciphertexts:
            split_ciphertext = pad_and_split_single_ciphertext(ciphertext)
            split_ciphertexts.extend(split_ciphertext)
    else:
        ciphertext = ciphertexts
        split_ciphertexts = pad_and_split_single_ciphertext(ciphertext)
    return split_ciphertexts, input_length


def average_predictions(predictions):
    """Calculate average prediction over `predictions`"""
    prediction = predictions[0]
    for res in predictions[1:]:
        prediction = np.add(prediction, res)
    return np.divide(prediction, len(predictions))


def predict_ffnn(model, inputs, batch_size, backend):
    result = []
    if backend == Backend.KERAS:
        result = model.predict(tf.convert_to_tensor(inputs), batch_size, verbose=0)
    elif backend == Backend.PYTORCH:
        if isinstance(inputs, tf.Tensor):
            inputs = inputs.numpy()
        input = torch.tensor(inputs, dtype=torch.float32, device=get_pytorch_device())
        result = model.predict(input, batch_size).cpu()
    else:
        raise ValueError(f"Unsupported backend: {backend} for FFNN architecture")
    return result


def predict_lstm(model, ciphertext, batch_size, backend):
    split_ciphertext, _ = split_ciphertext_to_model_input_length(
        model, "LSTM", ciphertext
    )

    predictions = []
    if backend == Backend.KERAS:
        for ct in split_ciphertext:
            predictions.append(
                model.predict(tf.convert_to_tensor([ct]), batch_size, verbose=0)
            )
    elif backend == Backend.PYTORCH:
        for ct in split_ciphertext:
            input = torch.tensor([ct], dtype=torch.long, device=get_pytorch_device())
            predictions.append(model.predict(input, batch_size).cpu())
    else:
        raise ValueError(f"Unsupported backend: {backend} for LSTM architecture")

    return average_predictions(predictions)


def predict_cnn(model, ciphertext, batch_size):
    split_ciphertext, input_length = split_ciphertext_to_model_input_length(
        model, "CNN", ciphertext
    )

    predictions = []
    for ct in split_ciphertext:
        reshaped_input = tf.reshape(tf.convert_to_tensor([ct]), (1, input_length, 1))
        prediction = model.predict(reshaped_input, batch_size, verbose=0)
        predictions.append(prediction)

    return average_predictions(predictions)


def predict_transformer(model, ciphertext, batch_size):
    split_ciphertext, _ = split_ciphertext_to_model_input_length(
        model, "Transformer", ciphertext
    )

    predictions = []
    for ct in split_ciphertext:
        predictions.append(
            model.predict(tf.convert_to_tensor([ct]), batch_size, verbose=0)
        )

    return average_predictions(predictions)
