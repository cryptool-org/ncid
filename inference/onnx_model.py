"""ONNX model adapter speaking the framework-independent prediction protocol.

The ONNX graphs exported by `util/export_models_to_onnx.py` have a static
batch dimension of 1, so `predict` runs the samples one at a time. The
prediction semantics match the training-side path:

- statistics models (FFNN, RF, NB) consume the 724-dim feature vector
- token models (LSTM, Transformer) consume the 1000-token numberspace
  sequence, fed as float32 (the Keras graphs cast internally to int for the
  embedding, as before)
- the rotor model (Rotor-SVM) consumes the 144-dim rotor statistics

For the scikit-learn exports the ONNX graph exposes a ZipMap output named
`output_probability` mapping class label to probability; the rows are built
with the class labels in sorted order, which matches the column order of
sklearn's `predict_proba` (classes_ sorted).
"""

import numpy as np

from cipherTypeDetection.preprocessing import pad_sequences


class OnnxModel:
    """Wraps an ONNX model file with the prediction protocol
    (`predict(statistics, ciphertexts, batch_size, verbose)`), the
    scikit-learn style `predict_proba` and the no-op `load_model` so that it
    can be passed anywhere the ensembles expect loadable model adapters.
    """

    STATISTICS = "statistics"
    TOKENS = "tokens"
    ROTOR_STATISTICS = "rotor_statistics"

    def __init__(self, onnx_path, architecture, input_kind):
        """Initialize an OnnxModel.

        Parameters
        ----------
        onnx_path : str
            Path to the `.onnx` model file.
        architecture : str
            The architecture name, e.g. "FFNN", "LSTM", "Transformer", "RF",
            "NB" or "Rotor-SVM".
        input_kind : str
            The kind of input the model consumes: "statistics" (feature
            engineering models), "tokens" (feature learning models) or
            "rotor_statistics" (the rotor only model).
        """
        if input_kind not in (self.STATISTICS, self.TOKENS, self.ROTOR_STATISTICS):
            raise ValueError(f"Unknown input kind: {input_kind}")
        self.onnx_path = onnx_path
        self.architecture = architecture
        self.input_kind = input_kind
        self._session = None

    def _get_session(self):
        """Lazily create the ONNX Runtime session."""
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.onnx_path)
        return self._session

    def load_model(self):
        """No-op for compatibility with ensembles that call `load_model` on
        their members. Returns `self`."""
        return self

    def predict(self, statistics, ciphertexts, batch_size=128, verbose=0):
        """Predict the cipher class probabilities via the prediction protocol.

        Parameters
        ----------
        statistics : list
            The feature engineering (or rotor) statistics of the ciphertexts.
        ciphertexts : list
            The ciphertexts in numberspace, used by the token models.
        batch_size : int
            Accepted for protocol compatibility; the ONNX graphs have a static
            batch dimension of 1 and are run per sample.
        verbose : int
            Accepted for protocol compatibility.

        Returns
        -------
        numpy.ndarray
            An array of shape [n, n_classes] with the predicted probabilities.
        """
        session = self._get_session()
        if self.input_kind in (self.STATISTICS, self.ROTOR_STATISTICS):
            rows = [np.asarray([statistics[i]], dtype=np.float32) for i in range(len(statistics))]
        elif self.input_kind == self.TOKENS:
            input_length = session.get_inputs()[0].shape[1]
            rows = []
            for ciphertext in ciphertexts:
                ciphertext = np.asarray(ciphertext, dtype=np.float32)
                if len(ciphertext) < input_length:
                    # self-tiling pad as the keras path did
                    ciphertext = pad_sequences([list(ciphertext)], maxlen=input_length)[0]
                elif len(ciphertext) > input_length:
                    ciphertext = ciphertext[:input_length]
                rows.append(np.asarray([ciphertext], dtype=np.float32))
        else:
            raise ValueError(f"Unknown input kind: {self.input_kind}")

        return np.vstack([self._run_single(session, row) for row in rows])

    def predict_proba(self, statistics):
        """sklearn-compatible probabilities of shape [n, n_classes_of_model].

        For the Rotor-SVM the 5 columns are ordered by class label (56..60),
        matching the column order of sklearn's `predict_proba` (classes_
        sorted). For the 61-class models the columns are 0..60.
        """
        if self.input_kind == self.TOKENS:
            raise ValueError("predict_proba is not supported for token-based models")
        session = self._get_session()
        rows = [np.asarray([statistic], dtype=np.float32) for statistic in statistics]
        return np.vstack([self._run_single(session, row) for row in rows])

    def _run_single(self, session, single_input):
        """Run the session for a single sample (static batch dim 1) and
        normalize the output to a float probability row."""
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        outputs = session.run(None, {input_name: single_input})
        for name, output in zip(output_names, outputs):
            if name == "output_probability":
                # ZipMap output: list of dicts {class_int: probability}. Build
                # the rows sorted by class label.
                rows = []
                for entry in output:
                    rows.append([float(entry[class_id]) for class_id in sorted(entry)])
                return np.asarray(rows, dtype=np.float32)
        # Raw tensor output: take the float output (NOT the argmax/label).
        for name, output in zip(output_names, outputs):
            output = np.asarray(output)
            if output.dtype.kind == "f":
                return output.astype(np.float32)
        raise ValueError(f"No usable output found in {self.onnx_path}")
