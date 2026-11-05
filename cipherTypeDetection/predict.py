"""Framework-free prediction of single ciphertext lines.

This module is the core of the inference code path: it maps a ciphertext line
to the model inputs (padded line, numberspace sequence, statistics) and calls
the model via the prediction protocol. A model speaks the protocol if it
provides

    predict(statistics, ciphertexts, batch_size, verbose) -> list of probability rows

The protocol is implemented by `cipherTypeDetection.models.modelFile.ModelFile`
(training-side wrappers) and by `inference.onnx_model.OnnxModel` (serving).
"""

from cipherImplementations.cipher import OUTPUT_ALPHABET, UNKNOWN_SYMBOL_NUMBER
from cipherTypeDetection.featureCalculations import calculate_statistics
from cipherTypeDetection.preprocessing import pad_line
from util.textutils import map_text_into_numberspace


def predict_ciphertext_line(ciphertext_bytes, model, ciphers, batch_size=128):
    """Predict the cipher type probabilities for a single line of ciphertext.

    Parameters
    ----------
    ciphertext_bytes : bytes
        A single line of ciphertext. A trailing newline is removed.
    model :
        A model implementing the prediction protocol, i.e. it provides
        `predict(statistics, ciphertexts, batch_size, verbose)`.
    ciphers : list
        The names of the ciphers the returned probabilities are restricted to.
    batch_size : int
        Batch size handed to the underlying model.

    Returns
    -------
    dict
        A dict mapping cipher name to probability in percent for the first
        `len(ciphers)` classes of the model. Empty lines and lines whose
        statistics cannot be calculated (ZeroDivisionError) yield an empty
        dict.
    """
    # remove newline
    line = ciphertext_bytes.strip(b'\n')
    if line == b'':
        return {}

    # Append ciphertext to itself. This improves the reliablity of the results.
    line = pad_line(line)
    # Limit line to at most 1000 characters to limit the execution time
    # (done inside pad_line)

    ciphertext = map_text_into_numberspace(line, OUTPUT_ALPHABET, UNKNOWN_SYMBOL_NUMBER)
    try:
        statistics = calculate_statistics(ciphertext)
    except ZeroDivisionError:
        return {}

    result = model.predict([statistics], [ciphertext], batch_size, verbose=0)

    # return a list of probabilities (does only return the last one in case a file is used)
    res_dict = {}
    if len(result) != 0:
        for j, val in enumerate(result[0]):
            if j >= len(ciphers):
                # Stop in cases where model is trained to predict more classes then are requested
                break
            res_dict[ciphers[j]] = val * 100
    return res_dict
