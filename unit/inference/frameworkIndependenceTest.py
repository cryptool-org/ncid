"""Guardrail tests for the framework-free serving path.

The API (`api.py`) and the inference package must not import any training
framework (TensorFlow, PyTorch, scikit-learn, pandas), so that the API can be
served from a minimal environment (see `requirements-api.txt`).

These tests are part of the regular unit test discovery:

    python3 -m unittest discover -s unit -p '*Test.py'
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level module prefixes that must not be imported by the serving path.
FORBIDDEN_MODULE_PREFIXES = (
    "tensorflow",
    "tf2onnx",
    "torch",
    "sklearn",
    "skl2onnx",
    "pandas",
)

# The modules that make up the serving path (import graph of api.py).
SERVING_MODULES = [
    "api.py",
    "inference/__init__.py",
    "inference/onnx_model.py",
    "inference/registry.py",
    "cipherTypeDetection/predict.py",
    "cipherTypeDetection/preprocessing.py",
    "cipherTypeDetection/constants.py",
    "cipherTypeDetection/ensembleModel.py",
    "cipherTypeDetection/rotorDifferentiationEnsemble.py",
    "cipherTypeDetection/featureCalculations.py",
    "cipherImplementations/cipher.py",
    "util/textutils.py",
]


def iter_imported_top_level_modules(source):
    """Yield the top-level module names of every import statement in `source`
    (including imports nested in functions)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module.split(".")[0]


class FrameworkIndependenceTest(unittest.TestCase):

    def test_serving_modules_do_not_import_training_frameworks(self):
        """Statically check that no serving module imports a training
        framework, at any nesting level. This check works in every
        environment, with or without the frameworks installed."""
        for relative_path in SERVING_MODULES:
            path = REPO_ROOT / relative_path
            self.assertTrue(path.exists(), f"missing serving module: {relative_path}")
            source = path.read_text()
            for module in iter_imported_top_level_modules(source):
                self.assertFalse(
                    module.startswith(FORBIDDEN_MODULE_PREFIXES),
                    f"{relative_path} imports the training framework module "
                    f"'{module}' (the serving path must be framework-free)")

    def test_api_imports_without_loading_training_frameworks(self):
        """Import `api` in a clean subprocess and assert that no training
        framework module ended up in sys.modules, even if the frameworks are
        installed in the environment."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi is not installed in this environment")

        code = (
            "import api, sys; "
            "prefixes = " + repr(FORBIDDEN_MODULE_PREFIXES) + "; "
            "tainted = [m for m in sys.modules "
            "if m.split('.')[0].startswith(prefixes)]; "
            "assert not tainted, tainted; "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"importing api loaded training frameworks:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
