"""Test the ciphertext normalization of the API.

The models operate on the lowercase output alphabet. Ciphertexts are
commonly written in uppercase (ACA convention), so the API normalizes the
case. Characters that cannot be mapped into the output alphabet are
rejected with a human-readable error instead of crashing the feature
engineering (regression test: uppercase input used to escape as a 500).

    python3 -m unittest discover -s unit/inference -p '*Test.py'
"""

import unittest

try:
    import api
    HAVE_API = True
except ImportError:
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "serving dependencies (fastapi, ...) required")
class CiphertextNormalizationTest(unittest.TestCase):
    def test_uppercase_is_normalized(self):
        self.assertEqual(api.normalize_and_validate_ciphertext("ABCD"), "abcd")

    def test_mixed_case_is_normalized(self):
        self.assertEqual(api.normalize_and_validate_ciphertext("aBcD eFgH"), "abcd efgh")

    def test_valid_input_is_unchanged(self):
        text = "yingraobhoarhthosortmkicwhaslcbpirpocuedcfthcez 12345 #"
        self.assertEqual(api.normalize_and_validate_ciphertext(text), text)

    def test_unsupported_characters_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            api.normalize_and_validate_ciphertext("hello, world!")
        self.assertIn("unsupported characters", str(ctx.exception))
        self.assertIn("','", str(ctx.exception))
        self.assertIn("'!'", str(ctx.exception))
        self.assertIn("a-z", str(ctx.exception))

    def test_unicode_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            api.normalize_and_validate_ciphertext("kryptölogie")
        self.assertIn("'ö'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
