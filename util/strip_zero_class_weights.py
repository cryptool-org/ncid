#!/usr/bin/env python3
"""Strip zero-weight class entries from ONNX TreeEnsembleClassifier graphs.

scikit-learn tree ensembles store one weight per (leaf, class) pair, even for
classes the leaf never predicts. For the NCID random forest 95.5% of these
entries are exactly zero. Since TreeEnsembleClassifier sums the weights of
all (leaf, class) entries of the reached leaf, zero-weight entries contribute
nothing to the prediction: dropping them (and their class_treeids /
class_nodeids / class_ids siblings) is a mathematically lossless
re-serialization with a dramatic effect on the model size at rest, the
ONNX Runtime load peak and the resident memory:

    rf_var_1000000.onnx:  1.86 GB ->  252 MB   (load peak 20.3 GB -> 2.8 GB)
    predictions:          bit-identical (verified against the original model)

The same stripping is applied automatically by
`util/export_models_to_onnx.py` when exporting scikit-learn models. This tool
post-processes an existing ONNX artifact:

    python3 util/strip_zero_class_weights.py --model data/models/rf.onnx
    python3 util/strip_zero_class_weights.py --model rf.onnx --output rf_small.onnx
"""

import argparse
import os

import onnx


def strip_zero_class_weights(model):
    """Drop (leaf, class) entries with weight 0.0 from all
    TreeEnsembleClassifier nodes of `model`, in place.

    Returns
    -------
    int
        The number of removed entries (0 if there was nothing to strip).
    """
    stripped = 0
    for node in model.graph.node:
        if node.op_type != "TreeEnsembleClassifier":
            continue
        attrs = {a.name: a for a in node.attribute}
        if "class_weights" not in attrs:
            continue
        weights = attrs["class_weights"].floats
        keep = [i for i, w in enumerate(weights) if w != 0.0]
        if len(keep) == len(weights):
            continue
        stripped += len(weights) - len(keep)
        for name in ("class_weights", "class_treeids", "class_nodeids", "class_ids"):
            attr = attrs[name]
            values = attr.floats if name == "class_weights" else attr.ints
            values[:] = [values[i] for i in keep]
    return stripped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Losslessly shrink ONNX TreeEnsembleClassifier models by "
                    "removing zero-weight class entries.")
    parser.add_argument("--model", required=True, help="Path to the input ONNX model.")
    parser.add_argument("--output", default=None,
                        help="Output path. Defaults to overwriting the input file.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file does not exist: {args.model}")
    output = args.output if args.output else args.model

    model = onnx.load(args.model)
    old_size = os.path.getsize(args.model)
    stripped = strip_zero_class_weights(model)
    if stripped == 0:
        print("No zero-weight class entries found; model left unchanged.")
        return
    onnx.save(model, output)
    new_size = os.path.getsize(output)
    print(f"Stripped {stripped:,} zero-weight class entries "
          f"({old_size / 1e6:.0f} MB -> {new_size / 1e6:.0f} MB): {output}")


if __name__ == "__main__":
    main()
