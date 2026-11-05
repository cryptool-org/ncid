"""Test the lossless stripping of zero-weight class entries from
TreeEnsembleClassifier graphs.

A tiny handcrafted forest with zero weights is stripped and must produce
bit-identical predictions through ONNX Runtime.

    python3 -m unittest discover -s unit/export -p '*Test.py'
"""

import tempfile
import unittest
from pathlib import Path

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False

try:
    import onnxruntime
    import numpy as np
    HAVE_RUNTIME = True
except ImportError:
    HAVE_RUNTIME = False


def build_tiny_forest(path):
    """Two one-branch trees over 3 classes with plenty of zero weights.

    tree 0: x[0] <= 0.5 -> leaf 1 (c0: 0.0, c1: 2.0, c2: 0.0)
                       else -> leaf 2 (c0: 1.0, c1: 0.0, c2: 0.0)
    tree 1: x[1] <= 0.5 -> leaf 1 (c0: 0.0, c1: 0.0, c2: 3.0)
                       else -> leaf 2 (c0: 0.0, c1: 1.0, c2: 0.0)
    """
    node = helper.make_node(
        "TreeEnsembleClassifier",
        ["input"],
        ["label", "probabilities"],
        domain="ai.onnx.ml",
        name="forest",
        nodes_treeids=[0, 0, 0, 1, 1, 1],
        nodes_nodeids=[0, 1, 2, 0, 1, 2],
        nodes_featureids=[0, 0, 0, 1, 1, 1],
        nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF"],
        nodes_truenodeids=[1, 0, 0, 1, 0, 0],
        nodes_falsenodeids=[2, 0, 0, 2, 0, 0],
        nodes_values=[0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
        class_treeids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        class_nodeids=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
        class_ids=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        class_weights=[0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0],
        classlabels_int64s=[0, 1, 2],
        post_transform="NONE",
    )
    graph = helper.make_graph(
        [node], "tiny_forest",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [1]),
         helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 3])])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13),
                              helper.make_opsetid("ai.onnx.ml", 1)])
    model.ir_version = 8
    onnx.save(model, path)


@unittest.skipUnless(HAVE_ONNX and HAVE_RUNTIME,
                     "onnx and onnxruntime are required")
class StripZeroClassWeightsTest(unittest.TestCase):
    def setUp(self):
        from util.strip_zero_class_weights import strip_zero_class_weights

        self.strip = strip_zero_class_weights
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.tmpdir.name) / "tiny.onnx"
        build_tiny_forest(self.model_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _predict(self, path):
        session = onnxruntime.InferenceSession(str(path))
        outputs = session.run(None, {"input": np.array([[0.3, 0.7]], dtype=np.float32)})
        return outputs[1]

    def test_strips_zero_entries(self):
        model = onnx.load(self.model_path)
        stripped = self.strip(model)
        self.assertEqual(stripped, 8)  # 12 class entries, 4 non-zero
        attr = {a.name: a for n in model.graph.node for a in n.attribute}
        self.assertEqual(len(attr["class_weights"].floats), 4)
        self.assertEqual(len(attr["class_ids"].ints), 4)
        self.assertEqual(len(attr["class_nodeids"].ints), 4)
        self.assertEqual(len(attr["class_treeids"].ints), 4)

    def test_predictions_are_bit_identical(self):
        reference = self._predict(self.model_path)
        model = onnx.load(self.model_path)
        self.strip(model)
        stripped_path = self.model_path.with_name("tiny_stripped.onnx")
        onnx.save(model, stripped_path)
        stripped = self._predict(stripped_path)
        np.testing.assert_array_equal(stripped, reference)

    def test_idempotent(self):
        model = onnx.load(self.model_path)
        self.strip(model)
        self.assertEqual(self.strip(model), 0)


if __name__ == "__main__":
    unittest.main()
