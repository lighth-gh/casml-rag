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

import numpy as np

from casml_b0.artifacts import digest, file_hash, load_config, read_json, read_jsonl, write_json
from casml_b0.contracts import load_queries
from casml_b0.context import pack_context, messages_for
from casml_b0.exporting import export, references_for
from casml_b0.generation import generate, generate_answer
from casml_b0.indexing import build_index
from casml_b0.llm import ExtractiveDemoGenerator
from casml_b0.prepare import prepare
from casml_b0.retrieval import BM25Index, reciprocal_rank_fusion, retrieve
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
        diagnosis = read_json(self.base / "run/diagnosis.json")
        manifest = read_json(self.base / "run/manifest.json")
        self.assertEqual(diagnosis["state"], "completed")
        self.assertEqual(diagnosis["run_id"], manifest["artifact_id"])
        self.assertTrue(diagnosis["summary"]["generation_ready_for_export"])
        self.assertEqual(manifest["files"]["diagnosis.json"], file_hash(self.base / "run/diagnosis.json"))
        predictions = read_jsonl(self.base / "run/predictions.jsonl")
        for query, prediction in zip(diagnosis["queries"], predictions):
            self.assertEqual(query["answer"], prediction["answer"])
            self.assertEqual(query["evidence"], prediction["evidence"])
            self.assertEqual(query["messages"], prediction["messages"])
        checksum = file_hash(self.base / "run/diagnosis.json")
        generate(self.cache, self.gen, self.config_path, self.base / "run")
        self.assertEqual(checksum, file_hash(self.base / "run/diagnosis.json"))

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
        diagnosis = read_json(self.base / "resume/diagnosis.json")
        self.assertEqual(diagnosis["state"], "failed")
        self.assertEqual(diagnosis["summary"]["errors"], ["Q002"])
        self.assertEqual(diagnosis["summary"]["unfinished"], ["Q003"])
        self.assertIn("simulated runtime interruption", diagnosis["queries"][1]["traceback"])
        checksum = file_hash(first)
        generate(self.cache, self.gen, self.config_path, self.base / "resume")
        self.assertEqual(checksum, file_hash(first))
        report = read_json(self.base / "resume/report.json")
        self.assertEqual(report["resumed"], 1)
        self.assertEqual(report["generated_this_call"], 2)
        diagnosis = read_json(self.base / "resume/diagnosis.json")
        self.assertEqual(diagnosis["state"], "completed")
        self.assertEqual(diagnosis["summary"]["resumed"], 1)
        self.assertEqual(diagnosis["summary"]["errors"], [])
        self.assertIsNone(diagnosis["failure"])

    def test_diagnosis_survives_model_loading_failure(self):
        with patch("casml_b0.generation.make_generator", side_effect=RuntimeError("model unavailable")):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                generate(self.cache, self.gen, self.config_path, self.base / "model_failure")
        diagnosis = read_json(self.base / "model_failure/diagnosis.json")
        self.assertEqual(diagnosis["state"], "failed")
        self.assertEqual(diagnosis["failure"]["type"], "RuntimeError")
        self.assertIn("model unavailable", diagnosis["failure"]["traceback"])
        self.assertEqual(diagnosis["summary"]["unfinished"], ["Q001", "Q002", "Q003"])
        self.assertFalse(diagnosis["summary"]["generation_ready_for_export"])

    def test_diagnosis_preserves_attempts_when_retry_errors_or_is_interrupted(self):
        for failure in (RuntimeError("retry failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                backend = ExtractiveDemoGenerator()
                run = self.base / type(failure).__name__
                first = {"answer": "unfinished first answer", "output_tokens": 100, "finish_reason": "length"}
                with patch.object(backend, "generate", side_effect=[first, failure]), \
                        patch("casml_b0.generation.make_generator", return_value=backend):
                    with self.assertRaises(type(failure)):
                        generate(self.cache, {**self.gen, "retry_max_new_tokens": 200}, self.config_path, run)
                diagnosis = read_json(run / "diagnosis.json")
                self.assertEqual(diagnosis["state"], "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")
                self.assertEqual(diagnosis["queries"][0]["answer"], first["answer"])
                self.assertEqual(len(diagnosis["queries"][0]["generation_attempts"]), 1)
                self.assertEqual(diagnosis["failure"]["type"], type(failure).__name__)
                self.assertFalse(diagnosis["summary"]["generation_ready_for_export"])

    def test_diagnosis_captures_all_errors_with_fail_fast_disabled(self):
        backend = ExtractiveDemoGenerator()
        with patch.object(backend, "generate", side_effect=RuntimeError("generation failed")), \
                patch("casml_b0.generation.make_generator", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "Some queries failed"):
                generate(self.cache, {**self.gen, "fail_fast": False}, self.config_path, self.base / "all_errors")
        diagnosis = read_json(self.base / "all_errors/diagnosis.json")
        self.assertEqual(diagnosis["state"], "failed")
        self.assertEqual(diagnosis["summary"]["errors"], ["Q001", "Q002", "Q003"])
        self.assertEqual(diagnosis["summary"]["unfinished"], [])

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
        report = read_json(self.base / "cutoff/report.json")
        detail = report["length_limited_details"][0]
        self.assertEqual(detail["query_id"], "Q001")
        self.assertEqual(detail["answer_tail"], "unfinished")
        self.assertTrue(detail["question"])
        diagnosis = read_json(self.base / "cutoff/diagnosis.json")
        self.assertEqual(diagnosis["state"], "completed")
        self.assertEqual(diagnosis["summary"]["length_limited"], ["Q001", "Q002", "Q003"])
        self.assertFalse(diagnosis["summary"]["generation_ready_for_export"])
        with self.assertRaisesRegex(ValueError, "Length-limited answers"):
            export(self.base / "cutoff", self.data / "queries.json", self.exp, self.base / "bad_cutoff")

    def test_production_guard_aborts_and_regenerates_length_limited_checkpoint(self):
        class NeverEnds(ExtractiveDemoGenerator):
            def __init__(self):
                self.calls = 0

            def generate(self, payload, config):
                self.calls += 1
                return {"answer": "unfinished", "output_tokens": config["max_new_tokens"],
                        "finish_reason": "length"}

        cfg = {**self.gen, "max_new_tokens": 100, "retry_max_new_tokens": 100,
               "retry_repetition_penalty": 1.1,
               "abort_after_consecutive_length_limited": 1}
        run = self.base / "length_guard"
        first = NeverEnds()
        with patch("casml_b0.generation.make_generator", return_value=first):
            with self.assertRaisesRegex(RuntimeError, "Aborting after 1 consecutive"):
                generate(self.cache, cfg, self.config_path, run)
        self.assertEqual(first.calls, 2)
        diagnosis = read_json(run / "diagnosis.json")
        self.assertEqual(diagnosis["state"], "failed")
        self.assertEqual(diagnosis["summary"]["length_limited"], ["Q001"])
        self.assertEqual(diagnosis["summary"]["unfinished"], ["Q002", "Q003"])

        second = NeverEnds()
        with patch("casml_b0.generation.make_generator", return_value=second):
            with self.assertRaisesRegex(RuntimeError, "Aborting after 1 consecutive"):
                generate(self.cache, cfg, self.config_path, run)
        self.assertEqual(second.calls, 2)
        self.assertEqual(read_json(run / "diagnosis.json")["summary"]["resumed"], 0)

    def test_concise_retry_changes_decoding_and_records_actual_prompt(self):
        class RepeatsUntilPolicyChanges(ExtractiveDemoGenerator):
            def __init__(self):
                self.calls = []

            def generate(self, payload, config):
                self.calls.append((payload, dict(config)))
                if (config.get("repetition_penalty", 1.0) <= 1.0 or
                        config.get("no_repeat_ngram_size", 0) == 0 or
                        "at most 180 words" not in payload["messages"][-1]["content"]):
                    return {"answer": "Repeated text. " * 100, "output_tokens": config["max_new_tokens"],
                            "finish_reason": "length"}
                return super().generate(payload, config)

        production = load_config(ROOT / "configs/generate.yaml")
        cfg = {**self.gen, **{k: v for k, v in production.items() if k.startswith("retry_")},
               "max_new_tokens": 512, "retry_max_new_tokens": 2048, "context_window": 4096}
        backend = RepeatsUntilPolicyChanges()
        with patch("casml_b0.generation.make_generator", return_value=backend):
            generate(self.cache, cfg, self.config_path, self.base / "concise")
        predictions = read_jsonl(self.base / "concise/predictions.jsonl")
        self.assertEqual(len(backend.calls), 6)
        for n, p in enumerate(predictions):
            first, retry = backend.calls[2*n][0], backend.calls[2*n + 1][0]
            self.assertNotIn("180 words", first["messages"][-1]["content"])
            self.assertEqual(first["evidence"], retry["evidence"])
            self.assertEqual(first["context"], retry["context"])
            self.assertIn(p["question"], retry["messages"][-1]["content"])
            self.assertEqual(p["messages"], retry["messages"])
            self.assertEqual(p["input_tokens"], backend.count_messages(p["messages"]))
            self.assertLessEqual(p["input_tokens"] + p["max_new_tokens_used"], cfg["context_window"])
            self.assertEqual(p["generation_attempts"][1]["no_repeat_ngram_size"], 8)
        export(self.base / "concise", self.data / "queries.json", self.exp, self.base / "concise_export")
        self.assertEqual(len(csv_rows(self.base / "concise_export/submission.csv")), 3)

    def test_retry_prompt_budget_is_recounted_at_context_limit(self):
        class TightWindow(ExtractiveDemoGenerator):
            context_window = 550

            def count_messages(self, messages):
                return 60

            def generate(self, payload, config):
                if payload["input_tokens"] + config["max_new_tokens"] > self.context_window:
                    raise AssertionError("context exceeded")
                return {"answer": "unfinished", "output_tokens": config["max_new_tokens"],
                        "finish_reason": "length"}

        payload = {"input_tokens": 38, "messages": [{"role": "user", "content": "Question"}]}
        cfg = {"max_new_tokens": 512, "retry_max_new_tokens": 2048, "context_window": 4096,
               "retry_instruction": "Be concise."}
        result = generate_answer(TightWindow(), payload, cfg)
        self.assertEqual([a["max_new_tokens"] for a in result["generation_attempts"]], [512, 490])
        self.assertEqual(result["input_tokens"], 60)
        self.assertEqual(payload["messages"][0]["content"], "Question")
        self.assertEqual(result["finish_reason"], "length")

    def test_retry_prompt_that_cannot_fit_keeps_original_metadata(self):
        backend = ExtractiveDemoGenerator()
        payload = {"input_tokens": 38, "messages": [{"role": "user", "content": "Question"}]}
        cfg = {"max_new_tokens": 512, "retry_max_new_tokens": 2048, "context_window": 550,
               "retry_instruction": "word " * 600}
        with patch.object(backend, "generate", return_value={"answer": "unfinished", "output_tokens": 512,
                                                            "finish_reason": "length"}):
            result = generate_answer(backend, payload, cfg)
        self.assertEqual(len(result["generation_attempts"]), 1)
        self.assertEqual(result["messages"], payload["messages"])
        self.assertEqual(result["input_tokens"], 38)
        self.assertIn("no output token budget", result["retry_skipped"])

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
        self.assertFalse(read_json(self.base / "partial/diagnosis.json")["summary"]["generation_ready_for_export"])
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

    def test_production_export_requires_official_sample(self):
        generate(self.cache, self.gen, self.config_path, self.base / "sample_gate")
        with self.assertRaisesRegex(ValueError, "Official sample_submission.csv"):
            export(self.base / "sample_gate", self.data / "queries.json",
                   {**self.exp, "require_sample": True}, self.base / "missing_sample")

    def test_demo_cannot_accidentally_be_exported_as_real_b0(self):
        generate(self.cache, self.gen, self.config_path, self.base / "demo")
        with self.assertRaisesRegex(ValueError, "Demo backends"):
            export(self.base / "demo", self.data / "queries.json", {**self.exp, "allow_demo": False}, self.base / "notreal")

    def test_duplicate_query_ids_rejected(self):
        path = self.data / "duplicates.json"
        write_json(path, [{"query_id": 1, "question": "a"}, {"query_id": "1", "question": "b"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_queries(path)

    def test_bm25_and_rrf_preserve_exact_term_and_dense_candidates(self):
        bm25 = BM25Index([
            "client centered therapy Carl Rogers nondirective",
            "unrelated material about visual perception",
            "therapy in general",
        ])
        scores = bm25.scores("What is client-centered therapy?")
        self.assertEqual(int(np.argmax(scores)), 0)
        fused, values = reciprocal_rank_fusion([1, 0], [0, 2], dense_weight=0.5, bm25_weight=0.5)
        self.assertEqual(fused[0], 0)
        self.assertGreater(values[0], values[1])

    def test_hybrid_retrieval_records_component_and_reranker_scores(self):
        class FakeReranker:
            def predict(self, pairs, **kwargs):
                return np.asarray([text.casefold().count(question.split()[0].casefold())
                                   for question, text in pairs], dtype=np.float32)

        config = {"method": "hybrid", "top_k": 3, "dense_pool_k": 4,
                  "bm25_pool_k": 4, "fusion_top_k": 4,
                  "reranker_backend": "cross_encoder"}
        target = self.base / "hybrid"
        with patch("casml_b0.retrieval._make_reranker", return_value=FakeReranker()):
            retrieve(self.index, self.data / "queries.json", config, target)
        rows = read_jsonl(target / "retrieval.jsonl")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row["candidates"]) == 3 for row in rows))
        for candidate in rows[0]["candidates"]:
            self.assertIn("dense_score", candidate)
            self.assertIn("bm25_score", candidate)
            self.assertIn("fusion_score", candidate)
            self.assertIsNotNone(candidate["reranker_score"])


if __name__ == "__main__":
    unittest.main()
