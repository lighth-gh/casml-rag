import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import yaml

from scripts.build_notebooks import GENERATION_SETUP, STEPS, paths

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

    def test_full_notebook_can_generate_without_official_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "competition"
            data.mkdir()
            (data / "book.pdf").write_bytes(b"%PDF-placeholder")
            (data / "queries.json").write_text(json.dumps([
                {"query_id": "1", "question": "What is memory?"}
            ]), encoding="utf-8")
            result = self.run_paths(data, auto_pdf=True, require_pdf=True, auto_queries=True,
                                    auto_sample=True, require_sample=False)
            self.assertIsNone(result["SAMPLE"])

    def test_production_generation_has_bounded_length_and_eos_guard(self):
        config = yaml.safe_load((ROOT / "configs/generate.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["max_new_tokens"], 384)
        self.assertEqual(config["retry_max_new_tokens"], 384)
        self.assertEqual(config["eos_token_ids"], [151645, 151643])
        self.assertEqual(config["abort_after_consecutive_length_limited"], 1)
        self.assertEqual(config["attn_implementation"], "sdpa")

        notebook = json.loads((ROOT / "notebooks/CASML_R1_end_to_end.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("GEN_MAX_NEW_TOKENS = 384", source)
        self.assertIn('generation_config["max_new_tokens"] = GEN_MAX_NEW_TOKENS', source)
        self.assertIn('GEN_RETRY_MAX_NEW_TOKENS = 384', source)
        self.assertIn('generation_config.setdefault("eos_token_ids", [151645, 151643])', source)
        self.assertIn("REQUIRE_SAMPLE = False", source)
        self.assertIn("SKIPPED EXPORT", source)

    def test_repository_contains_one_end_to_end_notebook(self):
        notebooks = sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(notebooks, ["CASML_R1_end_to_end.ipynb"])

    def test_runtime_config_enables_retry_and_uses_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.run_paths(Path(directory) / "missing")
            namespace.update(ROOT=ROOT, BASE_GEN_CONFIG=ROOT / "configs/generate.yaml")
            exec(GENERATION_SETUP, namespace)
            config = yaml.safe_load(namespace["GEN_CONFIG"].read_text(encoding="utf-8"))
            self.assertEqual(config["max_new_tokens"], 384)
            self.assertEqual(config["retry_max_new_tokens"], 384)
            self.assertEqual(namespace["RUN"].name, "hybrid_r2_qwen15b_t384_eos_guard")
            self.assertEqual(config["retry_repetition_penalty"], 1.15)
            self.assertEqual(config["retry_no_repeat_ngram_size"], 8)
            self.assertEqual(config["eos_token_ids"], [151645, 151643])
            self.assertEqual(config["abort_after_consecutive_length_limited"], 1)
            self.assertIn("180 words", config["retry_instruction"])
            self.assertEqual(config["model_name"], "Qwen/Qwen2.5-1.5B-Instruct")
            self.assertTrue(Path(config["system_prompt_file"]).is_file())

    def test_stale_clone_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "casml_b0").mkdir()
            (root / "casml_b0/generation.py").write_text("# old retry_max_new_tokens code")
            with self.assertRaisesRegex(RuntimeError, "ROOT_OVERRIDE"):
                exec(GENERATION_SETUP, {"ROOT": root})

    def test_failed_generation_still_displays_diagnosis_link(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "diagnosis.json").write_text('{}')
            display_module = ModuleType("IPython.display")
            display_module.FileLink = Mock(side_effect=lambda path: path)
            display_module.display = Mock()
            namespace = {"stage": Mock(side_effect=RuntimeError("stage failed")), "RUN": run,
                         "RETRIEVAL": run, "GEN_CONFIG": run / "generate.yaml"}
            with patch.dict("sys.modules", {"IPython": ModuleType("IPython"), "IPython.display": display_module}):
                with self.assertRaisesRegex(RuntimeError, "stage failed"):
                    exec(STEPS[4][2], namespace)
            display_module.display.assert_called_once_with(str(run / "diagnosis.json"))

    def test_export_stage_skips_cleanly_without_official_sample(self):
        stage = Mock()
        exec(STEPS[5][2], {"stage": stage, "SAMPLE": None})
        stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
