"""Test loading zstd-compressed ONNX models.

The registry prefers `<model>.onnx.zst` over the plain `.onnx` file, so the
models can be stored compressed at rest (the RF artifact shrinks from
1.86 GB to ~40 MB). These tests use a tiny synthetic ONNX model so they do
not depend on the deployed model artifacts.

    python3 -m unittest discover -s unit/inference -p '*Test.py'
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False

try:
    import onnxruntime  # noqa: F401
    import zstandard
    HAVE_RUNTIME = True
except ImportError:
    HAVE_RUNTIME = False


def build_tiny_model(path):
    """Build a minimal float model: output = input + 1."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("Add", ["input", "ones"], ["output"])
    ones = numpy_helper.from_array(np.ones((1, 2), dtype=np.float32), "ones")
    graph = helper.make_graph([node], "tiny", [input_tensor], [output_tensor], [ones])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


@unittest.skipUnless(HAVE_ONNX and HAVE_RUNTIME,
                     "onnx, onnxruntime and zstandard are required")
class ZstdModelLoadingTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmpdir.name)
        self.onnx_path = self.model_dir / "tiny.onnx"
        self.zst_path = self.model_dir / "tiny.onnx.zst"
        build_tiny_model(self.onnx_path)
        with open(self.onnx_path, "rb") as f, open(self.zst_path, "wb") as g:
            g.write(zstandard.ZstdCompressor().compress(f.read()))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_onnx_model_loads_from_zst(self):
        from inference.onnx_model import OnnxModel
        model = OnnxModel(str(self.zst_path), "tiny", OnnxModel.STATISTICS)
        result = model.predict([np.array([1.0, 2.0], dtype=np.float32)],
                               None, batch_size=1)
        np.testing.assert_allclose(result, [[2.0, 3.0]], rtol=0)

    def test_registry_prefers_zst(self):
        import inference.registry as registry

        original = dict(registry.MODEL_REGISTRY)
        try:
            registry.MODEL_REGISTRY["tiny"] = ("tiny.onnx", "statistics")
            loaded = registry.load_model("tiny", str(self.model_dir))
            self.assertTrue(loaded.onnx_path.endswith(".onnx.zst"))
            # the compressed model must be served correctly
            result = loaded.predict([np.array([5.0, 5.0], dtype=np.float32)],
                                    None, batch_size=1)
            np.testing.assert_allclose(result, [[6.0, 6.0]], rtol=0)
        finally:
            registry.MODEL_REGISTRY.clear()
            registry.MODEL_REGISTRY.update(original)


if __name__ == "__main__":
    unittest.main()
