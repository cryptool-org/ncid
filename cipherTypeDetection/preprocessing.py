"""Framework-free input preprocessing for the prediction path.

These helpers prepare ciphertext lines and numberspace sequences for the
models. They are part of the inference code path and therefore must not import
any training framework.
"""

import numpy as np


# The model input length used for sequence models (LSTM, Transformer) and for
# the rotor statistics calculation.
MODEL_INPUT_LENGTH = 1000


def pad_sequences(sequences, maxlen):
    """Pad sequences with data from itself."""
    ret_sequences = []
    for sequence in sequences:
        length = len(sequence)
        sequence = sequence * (maxlen // length) + sequence[:maxlen % length]
        ret_sequences.append(sequence)
    return np.array(ret_sequences)


def pad_line(line, length=MODEL_INPUT_LENGTH):
    """Append the ciphertext line to itself until it reaches at least `length`
    characters, then limit it to at most `length` characters.

    Appending the line to itself improves the reliability of the prediction
    results; limiting the length limits the execution time.
    """
    while len(line) < length:
        line = line + line
    return line[:length]


def pad_ciphertext(ciphertext, length=MODEL_INPUT_LENGTH):
    """Numberspace variant of `pad_line`: append the sequence to itself until it
    reaches at least `length` elements, then limit it to at most `length`."""
    ciphertext = np.asarray(ciphertext)
    while len(ciphertext) < length:
        ciphertext = np.concatenate((ciphertext, ciphertext), axis=0)
    return ciphertext[:length]
