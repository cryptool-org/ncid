import pickle
import tensorflow as tf
import torch

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

from cipherTypeDetection import config
from cipherTypeDetection.config import Backend
from cipherTypeDetection.models.ffnn import FFNN
from cipherTypeDetection.models.lstm import LSTM
from cipherTypeDetection.transformer import (
    MultiHeadSelfAttention,
    TransformerBlock,
    TokenAndPositionEmbedding,
)
from util.utils import get_pytorch_device


class ModelFile:
    """Wraps the model instance with additional information."""

    def __init__(self, implementation, architecture, backend):
        """Initialize a ModelFile.

        Parameters:
        -----------
        implementation
            Either a path to the model file or the actual model implementation, i.e.
            a PyTorch or Keras model.
        architecture
            The architecture of the model.
        backend
            The `Backend` of the model. Either Backend.KERAS or Backend.PYTORCH
        """
        self.implementation = implementation
        self.architecture = architecture
        self.backend = backend

    def load_model(self):
        # If implementation is not a path, the model itself is saved in implemenation.
        if not isinstance(self.implementation, str):
            return self.implementation

        keras_architectures = (
            "FFNN",
            "CNN",
            "LSTM",
            "Transformer",
        )

        if self.backend == Backend.PYTORCH:
            return self._load_pytorch()
        elif self.backend == Backend.KERAS and self.architecture in keras_architectures:
            return self._load_keras()
        elif self.backend == Backend.SCIKIT:
            return self._load_scikit()
        else:
            raise ValueError(f"Unknown backend: {self.backend}!")

    def _load_pytorch(self):
        device = get_pytorch_device()
        checkpoint = torch.load(self.implementation, map_location=device)

        if self.architecture == "FFNN":
            model = FFNN(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                output_size=checkpoint["output_size"],
                num_hidden_layers=checkpoint["num_hidden_layers"],
            )
        elif self.architecture == "LSTM":
            model = LSTM(
                vocab_size=checkpoint["vocab_size"],
                embed_dim=checkpoint["embed_dim"],
                hidden_size=checkpoint["hidden_size"],
                output_size=checkpoint["output_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
            )
        else:
            raise ValueError(f"Unimplemented PyTorch architecutre: {self.architecture}")

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)
        return model

    def _load_keras(self):
        if self.architecture == "Transformer":
            model = tf.keras.models.load_model(
                self.implementation,
                custom_objects={
                    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                    "MultiHeadSelfAttention": MultiHeadSelfAttention,
                    "TransformerBlock": TransformerBlock,
                },
            )
        elif self.architecture in ("FFNN", "CNN", "LSTM"):
            model = tf.keras.models.load_model(self.implementation)
        else:
            raise ValueError(f"Unimplemented Keras architecture: {self.architecture}")
        optimizer = Adam(
            learning_rate=config.learning_rate,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
            epsilon=config.epsilon,
            amsgrad=config.amsgrad,
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy"),
            ],
        )
        return model

    def _load_scikit(self):
        with open(self.implementation, "rb") as f:
            return pickle.load(f)
