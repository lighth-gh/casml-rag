import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.build_notebooks import GENERATION_SETUP, paths

ROOT = Path(__file__).resolve().parents[1]


class NotebookInputDiscovery(unittest.TestCase):
    def run_paths(self, data, **options):
        source = "".join(paths(**options)["source"])
        source = source.replace("INPUT_ROOT_OVERRIDE = None", f"INPUT_ROOT_OVERRIDE = {str(data)!r}")
        namespace = {"ROOT": Path(data).parent, "WORK": Path(data).parent / "work"}
        exec(source, namespace)
        return namespace

    def test_discovers_actual_competition_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "competition"
            data.mkdir()
            book = data / "OpenStax Psychology 2e.pdf"
            book.write_bytes(b"%PDF-placeholder")
            queries = data / "queries.json"
            queries.write_text(json.dumps([
                {"query_id": "Q1", "question": "What is memory?"}
            ]), encoding="utf-8")
            sample = data / "sample_submission.csv"
            sample.write_text("ID,context,answer,references\nQ1,,,\n", encoding="utf-8")

            result = self.run_paths(
                data, auto_pdf=True, require_pdf=True, auto_queries=True, auto_sample=True
            )

            self.assertEqual(result["PDF"], book.resolve())
            self.assertEqual(result["QUERIES"], queries.resolve())
            self.assertEqual(result["SAMPLE"], sample.resolve())

    def test_missing_pdf_fails_before_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "competition"
            data.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "PDF_OVERRIDE"):
                self.run_paths(data, auto_pdf=True, require_pdf=True)

    def test_generation_stage_does_not_require_competition_files(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-attached"
            result = self.run_paths(missing)
            self.assertIsNone(result["PDF"])
            self.assertIsNone(result["QUERIES"])
            self.assertIsNone(result["SAMPLE"])

    def test_production_generation_limit_exceeds_observed_cutoff(self):
        config = yaml.safe_load((ROOT / "configs/generate.yaml").read_text(encoding="utf-8"))
        self.assertGreaterEqual(config["max_new_tokens"], 512)

        notebook = json.loads((ROOT / "notebooks/00_run_b0.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("GEN_MAX_NEW_TOKENS = 512", source)
        self.assertIn('generation_config["max_new_tokens"] = GEN_MAX_NEW_TOKENS', source)
        self.assertIn('GEN_RETRY_MAX_NEW_TOKENS = 2048', source)

    def test_runtime_config_enables_retry_and_uses_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.run_paths(Path(directory) / "missing")
            namespace.update(ROOT=ROOT, BASE_GEN_CONFIG=ROOT / "configs/generate.yaml")
            exec(GENERATION_SETUP, namespace)
            config = yaml.safe_load(namespace["GEN_CONFIG"].read_text(encoding="utf-8"))
            self.assertEqual(config["max_new_tokens"], 512)
            self.assertEqual(config["retry_max_new_tokens"], 2048)
            self.assertEqual(namespace["RUN"].name, "b0_g03_t512_r2048")
            self.assertEqual(config["retry_repetition_penalty"], 1.15)
            self.assertEqual(config["retry_no_repeat_ngram_size"], 8)
            self.assertIn("180 words", config["retry_instruction"])
            self.assertTrue(Path(config["system_prompt_file"]).is_file())

    def test_stale_clone_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "casml_b0").mkdir()
            (root / "casml_b0/generation.py").write_text("# old retry_max_new_tokens code")
            with self.assertRaisesRegex(RuntimeError, "ROOT_OVERRIDE"):
                exec(GENERATION_SETUP, {"ROOT": root})


if __name__ == "__main__":
    unittest.main()
