"""Verify the actual HF adapter's decoding arguments without downloading a model."""
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from casml_b0.llm import HFGenerator


class HFDecoding(unittest.TestCase):
    def backend(self, output, eos):
        backend = HFGenerator.__new__(HFGenerator)
        backend.device = "cpu"
        backend.torch = SimpleNamespace(inference_mode=nullcontext)
        backend.tokenizer = MagicMock()
        backend.tokenizer.pad_token_id = 2
        backend.pad_token_id = 2
        backend.eos_token_ids = eos if isinstance(eos, list) else [eos]
        backend.tokenizer.return_value.to.return_value = {"input_ids": SimpleNamespace(shape=(1, 10))}
        backend.tokenizer.decode.return_value = "Answer."
        backend.model = MagicMock()
        backend.model.generation_config.eos_token_id = eos
        backend.model.generate.return_value.__getitem__.return_value = output
        return backend

    def test_retry_controls_reach_model_generate(self):
        backend = self.backend([7, 2], [2, 3])
        result = backend.generate({"messages": [], "input_tokens": 10},
                                  {"max_new_tokens": 2, "repetition_penalty": 1.15, "no_repeat_ngram_size": 8})
        kwargs = backend.model.generate.call_args.kwargs
        self.assertEqual(kwargs["repetition_penalty"], 1.15)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 8)
        self.assertEqual(kwargs["eos_token_id"], [2, 3])
        self.assertEqual(kwargs["pad_token_id"], 2)
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(result["finish_reason"], "eos")

    def test_token_limit_without_eos_remains_length_limited(self):
        backend = self.backend([7, 8], 2)
        result = backend.generate({"messages": [], "input_tokens": 10}, {"max_new_tokens": 2})
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(backend.model.generate.call_args.kwargs["no_repeat_ngram_size"], 0)


if __name__ == "__main__":
    unittest.main()
