"""Contract and failure-path tests: no GPU or downloaded model required."""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casml_b0.artifacts import digest, file_hash, load_config, read_json, read_jsonl, write_json
from casml_b0.contracts import load_queries
from casml_b0.context import pack_context, messages_for
from casml_b0.exporting import export, references_for
from casml_b0.generation import generate, generate_answer
from casml_b0.indexing import build_index
from casml_b0.llm import ExtractiveDemoGenerator
from casml_b0.prepare import prepare
from casml_b0.retrieval import retrieve
from scripts.make_demo import create_demo

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        create_demo(self.data)
        self.corpus, self.index, self.cache = [self.base / p for p in ("corpus", "index", "retrieval")]
        prepare(self.data / "demo_book.pdf", load_config(ROOT / "configs/demo_prepare.yaml"), self.corpus,
                self.data / "page_map_override.csv")
        build_index(self.corpus, load_config(ROOT / "configs/demo_index.yaml"), self.index)
        retrieve(self.index, self.data / "queries.json", {"top_k": 20}, self.cache)
        self.gen = load_config(ROOT / "configs/demo_generate.yaml")
        self.config_path = ROOT / "configs/demo_generate.yaml"
        self.exp = load_config(ROOT / "configs/demo_export.yaml")

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end_csv_and_exact_source_spans(self):
        generate(self.cache, self.gen, self.config_path, self.base / "run")
        export(self.base / "run", self.data / "queries.json", self.exp, self.base / "export",
               sample=self.data / "sample_submission.csv", pdf=self.data / "demo_book.pdf")
        rows = csv_rows(self.base / "export/submission.csv")
        self.assertEqual([r["ID"] for r in rows], ["Q001", "Q002", "Q003"])
        pages = {p["pdf_page"]: p for p in read_jsonl(self.corpus / "pages.jsonl")}
        for c in read_jsonl(self.corpus / "chunks.jsonl"):
            self.assertEqual(c["text"], pages[c["pdf_page"]]["text"][c["char_start"]:c["char_end"]])
        for row in rows:
            self.assertIsInstance(json.loads(row["references"])["pages"], list)
        self.assertEqual(file_hash(self.data / "demo_book.pdf"), file_hash(self.base / "export/book.pdf"))

    def test_generation_without_corpus_index_or_retrieval_imports(self):
        shutil.rmtree(self.corpus)
        shutil.rmtree(self.index)
        script = f'''import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name in {{"embedding", "retrieval", "indexing"}} or name.startswith(("faiss", "sentence_transformers", "casml_b0.retrieval", "casml_b0.indexing", "casml_b0.embedding")):
        raise AssertionError("Generation touched retrieval dependency: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from casml_b0.generation import generate
from casml_b0.artifacts import load_config
generate({str(self.cache)!r}, load_config({str(self.config_path)!r}), {str(self.config_path)!r}, {str(self.base / 'standalone')!r})
'''
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True)
        self.assertTrue(read_json(self.base / "standalone/manifest.json")["complete"])

    def test_generation_config_change_preserves_retrieval_bytes(self):
        before = {p.name: file_hash(p) for p in self.cache.iterdir() if p.is_file()}
        generate(self.cache, self.gen, self.config_path, self.base / "run1")
        changed = {**self.gen, "context_top_k": 1, "max_new_tokens": 120}
        with self.assertRaisesRegex(ValueError, "NEW --out"):
            generate(self.cache, changed, self.config_path, self.base / "run1")
        generate(self.cache, changed, self.config_path, self.base / "run2")
        self.assertEqual(before, {p.name: file_hash(p) for p in self.cache.iterdir() if p.is_file()})
        self.assertNotEqual(read_json(self.base / "run1/manifest.json")["artifact_id"],
                            read_json(self.base / "run2/manifest.json")["artifact_id"])

    def test_prompt_content_change_cannot_mix_checkpoints(self):
        system, user = self.base / "system.txt", self.base / "user.txt"
        system.write_text("Answer from evidence only.")
        user.write_text("Question: {question}\nContext: {context}")
        cfg = {**self.gen, "system_prompt_file": str(system), "user_prompt_file": str(user)}
        generate(self.cache, cfg, self.config_path, self.base / "prompt_run")
        system.write_text("Answer briefly from evidence only.")
        with self.assertRaisesRegex(ValueError, "NEW --out"):
            generate(self.cache, cfg, self.config_path, self.base / "prompt_run")

    def test_resume_retries_error_but_keeps_successful_checkpoint(self):
        class FailSecond(ExtractiveDemoGenerator):
            def __init__(self):
                self.calls = 0
            def generate(self, payload, config):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated runtime interruption")
                return super().generate(payload, config)
        with patch("casml_b0.generation.make_generator", return_value=FailSecond()):
            with self.assertRaises(RuntimeError):
                generate(self.cache, self.gen, self.config_path, self.base / "resume")
        first = self.base / f"resume/checkpoints/{digest('Q001')}.json"
        checksum = file_hash(first)
        generate(self.cache, self.gen, self.config_path, self.base / "resume")
        self.assertEqual(checksum, file_hash(first))
        report = read_json(self.base / "resume/report.json")
        self.assertEqual(report["resumed"], 1)
        self.assertEqual(report["generated_this_call"], 2)

    def test_retrieval_tampering_is_detected_before_generation(self):
        with open(self.cache / "retrieval.jsonl", "a") as f:
            f.write("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            generate(self.cache, self.gen, self.config_path, self.base / "tampered")

    def test_length_retry_recovers_only_cutoff_query_and_exports(self):
        class CutoffSecond(ExtractiveDemoGenerator):
            def __init__(self):
                self.calls = []

            def generate(self, payload, config):
                self.calls.append((payload["messages"], config["max_new_tokens"]))
                # First query succeeds; second needs more than 1024 tokens.
                if len(self.calls) in (2, 3):
                    return {"answer": "unfinished", "output_tokens": config["max_new_tokens"],
                            "finish_reason": "length"}
                return super().generate(payload, config)

        backend = CutoffSecond()
        cfg = {**self.gen, "max_new_tokens": 512, "retry_max_new_tokens": 2048, "context_window": 4096}
        before = file_hash(self.cache / "retrieval.jsonl")
        with patch("casml_b0.generation.make_generator", return_value=backend):
            generate(self.cache, cfg, self.config_path, self.base / "retry")
        self.assertEqual([tokens for _, tokens in backend.calls], [512, 512, 1024, 2048, 512])
        self.assertEqual(backend.calls[1][0], backend.calls[2][0])
        self.assertEqual(backend.calls[1][0], backend.calls[3][0])
        self.assertEqual(before, file_hash(self.cache / "retrieval.jsonl"))
        report = read_json(self.base / "retry/report.json")
        self.assertEqual(report["retried_for_length"], ["Q002"])
        self.assertEqual(report["length_limited"], [])
        export(self.base / "retry", self.data / "queries.json", self.exp, self.base / "retry_export")
        self.assertEqual(len(csv_rows(self.base / "retry_export/submission.csv")), 3)

    def test_exhausted_length_retry_still_blocks_export(self):
        class NeverEnds(ExtractiveDemoGenerator):
            def generate(self, payload, config):
                return {"answer": "unfinished", "output_tokens": config["max_new_tokens"],
                        "finish_reason": "length"}

        cfg = {**self.gen, "max_new_tokens": 512, "retry_max_new_tokens": 2048, "context_window": 4096}
        with patch("casml_b0.generation.make_generator", return_value=NeverEnds()):
            generate(self.cache, cfg, self.config_path, self.base / "cutoff")
        predictions = read_jsonl(self.base / "cutoff/predictions.jsonl")
        self.assertTrue(all(len(p["generation_attempts"]) == 3 for p in predictions))
        with self.assertRaisesRegex(ValueError, "Length-limited answers"):
            export(self.base / "cutoff", self.data / "queries.json", self.exp, self.base / "bad_cutoff")

    def test_length_retry_respects_actual_model_context_window(self):
        class SmallWindow(ExtractiveDemoGenerator):
            context_window = 1000

            def generate(self, payload, config):
                self.assert_budget = payload["input_tokens"] + config["max_new_tokens"]
                if self.assert_budget > self.context_window:
                    raise AssertionError("model context exceeded")
                return {"answer": "unfinished", "output_tokens": config["max_new_tokens"],
                        "finish_reason": "length"}

        result = generate_answer(SmallWindow(), {"input_tokens": 400},
                                 {"max_new_tokens": 512, "retry_max_new_tokens": 2048, "context_window": 4096})
        self.assertEqual([a["max_new_tokens"] for a in result["generation_attempts"]], [512, 600])
        self.assertEqual(result["finish_reason"], "length")

    def test_invalid_retry_budget_fails_before_model_load(self):
        with patch("casml_b0.generation.make_generator") as factory:
            with self.assertRaisesRegex(ValueError, "retry_max_new_tokens"):
                generate(self.cache, {**self.gen, "max_new_tokens": 512, "retry_max_new_tokens": 256},
                         self.config_path, self.base / "invalid_retry")
            factory.assert_not_called()

    def test_context_budget_preserves_question_and_whole_chunks(self):
        row = read_jsonl(self.cache / "retrieval.jsonl")[0]
        backend = ExtractiveDemoGenerator()
        config = {**self.gen, "context_token_budget": 70, "context_top_k": 4}
        payload = pack_context(row, backend, config, "Use evidence", "Q: {question}\nC: {context}")
        self.assertIn(row["question"], payload["messages"][1]["content"])
        self.assertLessEqual(payload["context_tokens"], 70)
        self.assertLessEqual(payload["input_tokens"] + config["max_new_tokens"], config["context_window"])
        original = {c["chunk_id"]: c["text"] for c in row["candidates"]}
        for e in payload["evidence"]:
            self.assertEqual(e["text"], original[e["chunk_id"]])
        too_small = {**config, "context_window": 101}
        with self.assertRaisesRegex(ValueError, "NEVER truncated"):
            pack_context(row, backend, too_small, "Use evidence", "Q: {question}\nC: {context}")

    def test_references_use_only_packed_evidence_and_printed_mapping(self):
        cfg = {**self.gen, "context_top_k": 1}
        generate(self.cache, cfg, self.config_path, self.base / "one")
        export(self.base / "one", self.data / "queries.json", self.exp, self.base / "export_one")
        predictions = read_jsonl(self.base / "one/predictions.jsonl")
        rows = csv_rows(self.base / "export_one/submission.csv")
        for p, r in zip(predictions, rows):
            self.assertEqual(len(p["evidence"]), 1)
            self.assertEqual(json.loads(r["references"])["pages"], [str(p["evidence"][0]["printed_page"])])
            self.assertEqual(r["context"], p["context"])
        broken = dict(predictions[0]["evidence"][0], printed_page=None)
        with self.assertRaisesRegex(ValueError, "unknown"):
            references_for([broken], self.exp)

    def test_partial_run_cannot_export_full_submission(self):
        generate(self.cache, self.gen, self.config_path, self.base / "partial", limit=1)
        with self.assertRaisesRegex(ValueError, "ALL input"):
            export(self.base / "partial", self.data / "queries.json", self.exp, self.base / "bad")

    def test_sample_order_is_preserved_and_wrong_pdf_rejected(self):
        generate(self.cache, self.gen, self.config_path, self.base / "ordered")
        sample = self.data / "reverse.csv"
        sample.write_text('ID,context,answer,references\nQ003,,,\nQ002,,,\nQ001,,,\n')
        export(self.base / "ordered", self.data / "queries.json", self.exp, self.base / "reversed", sample=sample)
        rows = csv_rows(self.base / "reversed/submission.csv")
        self.assertEqual([r["ID"] for r in rows], ["Q003", "Q002", "Q001"])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            export(self.base / "ordered", self.data / "queries.json", self.exp, self.base / "wrongpdf", pdf=sample)

    def test_demo_cannot_accidentally_be_exported_as_real_b0(self):
        generate(self.cache, self.gen, self.config_path, self.base / "demo")
        with self.assertRaisesRegex(ValueError, "Demo backends"):
            export(self.base / "demo", self.data / "queries.json", {**self.exp, "allow_demo": False}, self.base / "notreal")

    def test_duplicate_query_ids_rejected(self):
        path = self.data / "duplicates.json"
        write_json(path, [{"query_id": 1, "question": "a"}, {"query_id": "1", "question": "b"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_queries(path)


if __name__ == "__main__":
    unittest.main()
