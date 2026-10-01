import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import yaml

from casml_b0.generation import load_prompts
from scripts.build_notebooks import GENERATION_SETUP, STEPS, paths

ROOT = Path(__file__).resolve().parents[1]


class NotebookInputDiscovery(unittest.TestCase):
    def run_paths(self, data, experiment="baseline", **options):
        source = "".join(paths(**options)["source"])
        source = source.replace("INPUT_ROOT_OVERRIDE = None", f"INPUT_ROOT_OVERRIDE = {str(data)!r}")
        source = source.replace('EXPERIMENT = "baseline"', f"EXPERIMENT = {experiment!r}")
        namespace = {"ROOT": ROOT, "WORK": Path(data).parent / "work"}
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
        self.assertNotIn("SKIPPED EXPORT", source)
        self.assertIn("if SAMPLE is not None:", source)

        export_config = yaml.safe_load((ROOT / "configs/export.yaml").read_text(encoding="utf-8"))
        self.assertEqual(export_config["page_value_type"], "integer")
        self.assertNotIn("require_sample", export_config)

    def test_repository_contains_one_end_to_end_notebook(self):
        notebooks = sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(notebooks, ["CASML_R1_end_to_end.ipynb"])

    def test_runtime_config_reports_grounding_and_uses_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.run_paths(Path(directory) / "missing")
            namespace.update(ROOT=ROOT, BASE_GEN_CONFIG=ROOT / "configs/generate.yaml",
                             BASE_EXPORT_CONFIG=ROOT / "configs/export.yaml",
                             EXPORT_CONFIG=namespace["WORK"] / "configs/export_documented_schema.yaml")
            exec(GENERATION_SETUP, namespace)
            config = yaml.safe_load(namespace["GEN_CONFIG"].read_text(encoding="utf-8"))
            self.assertEqual(config["max_new_tokens"], 384)
            self.assertEqual(config["retry_max_new_tokens"], 384)
            self.assertEqual(namespace["RUN"].name, "hybrid_r4_baseline_qwen15b_t384_report_only")
            self.assertEqual(config["retry_repetition_penalty"], 1.15)
            self.assertEqual(config["retry_no_repeat_ngram_size"], 8)
            self.assertEqual(config["eos_token_ids"], [151645, 151643])
            self.assertEqual(config["abort_after_consecutive_length_limited"], 1)
            self.assertIn("180 words", config["retry_instruction"])
            self.assertFalse(config["grounding_validator_enabled"])
            self.assertTrue(config["grounding_report_enabled"])
            self.assertEqual(config["grounding_validator_max_retries"], 0)
            self.assertEqual(config["grounding_retry_max_new_tokens"], 224)
            self.assertFalse(config["grounding_deterministic_repair"])
            self.assertEqual(config["model_name"], "Qwen/Qwen2.5-1.5B-Instruct")
            self.assertTrue(Path(config["system_prompt_file"]).is_file())
            export_config = yaml.safe_load(namespace["EXPORT_CONFIG"].read_text(encoding="utf-8"))
            self.assertNotIn("require_sample", export_config)
            self.assertEqual(export_config["page_value_type"], "integer")
            self.assertFalse(export_config["fail_on_unsupported_claims"])
            self.assertEqual(export_config["max_answer_words"], 140)

    def test_notebook_experiments_change_one_factor_and_keep_separate_artifacts(self):
        configs, prompts, namespaces = {}, {}, {}
        for experiment in ("baseline", "attribution", "top3", "top6"):
            with tempfile.TemporaryDirectory() as directory:
                namespace = self.run_paths(Path(directory) / "missing", experiment=experiment)
                exec(GENERATION_SETUP, namespace)
                config = yaml.safe_load(namespace["GEN_CONFIG"].read_text(encoding="utf-8"))
                configs[experiment] = config
                prompts[experiment] = load_prompts(config, namespace["GEN_CONFIG"])
                namespaces[experiment] = namespace
                self.assertFalse(config["grounding_validator_enabled"])
                self.assertTrue(config["grounding_report_enabled"])
                export_config = yaml.safe_load(namespace["EXPORT_CONFIG"].read_text(encoding="utf-8"))
                self.assertFalse(export_config["fail_on_unsupported_claims"])
                self.assertTrue(export_config["fail_on_truncation"])
        baseline = configs["baseline"]
        for experiment, allowed in (("attribution", {"system_prompt_file"}),
                                    ("top3", {"context_top_k"}), ("top6", {"context_top_k"})):
            changed = {key for key in set(baseline) | set(configs[experiment])
                       if baseline.get(key) != configs[experiment].get(key)}
            self.assertEqual(changed, allowed)
        self.assertEqual(configs["baseline"]["context_top_k"], 4)
        self.assertEqual(configs["top3"]["context_top_k"], 3)
        self.assertEqual(configs["top6"]["context_top_k"], 6)
        self.assertEqual(prompts["top3"], prompts["baseline"])
        self.assertEqual(prompts["top6"], prompts["baseline"])
        self.assertEqual(prompts["attribution"][1], prompts["baseline"][1])
        self.assertIn("Keep each fact attached", prompts["attribution"][0])
        self.assertNotEqual(prompts["attribution"][0], prompts["baseline"][0])
        for key in ("RUN_NAME", "BASE_GEN_CONFIG"):
            self.assertEqual(len({str(n[key]) for n in namespaces.values()}), 4)
        for key in ("GEN_CONFIG", "EXPORT_CONFIG"):
            self.assertEqual(len({n[key].name for n in namespaces.values()}), 4)

    def test_unknown_experiment_is_rejected_before_runtime_config(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "EXPERIMENT"):
                self.run_paths(Path(directory) / "missing", experiment="typo")

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

    def test_export_stage_does_not_gate_on_official_sample(self):
        source = STEPS[5][2]
        self.assertNotIn("if SAMPLE is None", source)
        self.assertNotIn("SKIPPED EXPORT", source)
        self.assertIn("stage(*args)", source)


if __name__ == "__main__":
    unittest.main()
