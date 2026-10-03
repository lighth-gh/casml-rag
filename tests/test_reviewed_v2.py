"""Source validation and artifact boundaries; fixtures are NOT real reviewed gold."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from casml_b0.artifacts import begin, digest, finish, load_artifact, read_json, read_jsonl, signature, write_json, write_jsonl
from casml_b0.evaluation import evaluate, paired_comparison, score_evaluation, score_row, select_model
from casml_b0.finetuning import finetune, selected_generation_config, training_schedule
from casml_b0.reviewed import attach_context, prepare_reviewed, review_digest, validate_annotations

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    chunks, annotations = [], []
    for n, split in enumerate(("train", "dev", "holdout"), 1):
        text = f"Concept {n} is a distinct process. Experiments investigate concept {n}."
        chunk = {"chunk_id": f"c{n}", "doc_id": "book", "source_pdf": "book.pdf", "pdf_page": n,
                 "printed_page": str(n), "section_path": [f"section{n}"], "section_method": "fixture",
                 "text": text, "text_sha256": digest(text), "char_start": 0, "char_end": len(text)}
        support = [{"chunk_id": f"c{n}", "quote": text}]
        answer = f"Researchers investigate the distinct process called concept {n}."
        row = {"query_id": f"q{n}", "question": f"What is concept {n}?", "answer": answer,
               "split": split, "split_group": split, "question_type": "definition",
               "evidence_ids": [f"c{n}"], "required_facts": [{"fact_id": "f1", "text": text, "supports": support}],
               "answer_claims": [{"text": answer, "supports": support}], "reference_sets": [[f"c{n}"]],
               "review_status": "approved", "reviewer": "UNIT TEST FIXTURE"}
        row["review_sha256"] = review_digest(row)
        chunks.append(chunk)
        annotations.append(row)
    return chunks, annotations


def dataset(folder):
    chunks, annotations = fixtures()
    corpus = folder / "corpus"
    m, _ = begin(corpus, signature("corpus", {}, {}, []))
    write_jsonl(corpus / "chunks.jsonl", chunks)
    finish(corpus, m, ["chunks.jsonl"])
    for row in annotations:
        row["source_corpus_id"] = m["artifact_id"]
        row["review_sha256"] = review_digest(row)
    path = folder / "annotations.jsonl"
    write_jsonl(path, annotations)
    out = folder / "reviewed"
    prepare_reviewed(corpus, path, {"min_train_examples": 1, "min_dev_examples": 1, "min_holdout_examples": 1}, out)
    return out


class Backend:
    context_window = 4096
    count_text = staticmethod(lambda text: len(text.split()))
    count_messages = staticmethod(lambda messages: 100)

    def generate(self, payload, config):
        return {"answer": "A claim.", "finish_reason": "eos", "output_tokens": 3}


def generation_config():
    return {"model_name": "unit-test-model", "backend": "huggingface", "revision": "fixed",
            "context_window": 4096, "max_new_tokens": 100, "context_top_k": 4, "context_token_budget": 1900,
            "system_prompt_file": str(ROOT / "prompts/system.txt"),
            "user_prompt_file": str(ROOT / "prompts/user.txt")}


def reviews(predictions, good=True):
    return [{"query_id": p["query_id"], "prediction_sha256": digest(p), "review_status": "approved",
             "reviewer": "UNIT TEST FIXTURE", "severe_error": False,
             "context_relevant_spans": [[0, len(p["context"])]],
             "claims": [{"text": p["answer"], "supported": good, "correct": good,
                         "fact_ids": ["f1"] if good else []}]} for p in predictions]


class ReviewedTests(unittest.TestCase):
    def test_bootstrap_keeps_shared_sources_together_despite_different_group_labels(self):
        chunks, rows = fixtures()
        extra = copy.deepcopy(rows[1])
        extra.update(query_id="paraphrase", question="How would you define the second concept?", split_group="another-label")
        extra["review_sha256"] = review_digest(extra)
        rows.append(extra)
        splits, _ = validate_annotations(rows, chunks, {})
        self.assertEqual(splits["dev"][0]["bootstrap_group"], splits["dev"][1]["bootstrap_group"])

    def test_paraphrases_are_accepted_but_pending_and_stale_labels_are_not(self):
        chunks, rows = fixtures()
        result, _ = validate_annotations(rows, chunks, dict(train=1, dev=1, holdout=1))
        self.assertEqual(result["train"][0]["evidence_coverage"], 1)
        rows[0]["review_status"] = "pending"
        with self.assertRaisesRegex(ValueError, "Need 1 approved train"):
            validate_annotations(rows, chunks, dict(train=1))
        rows[0]["review_status"] = "approved"
        rows[0]["answer"] += " Unreviewed edit."
        with self.assertRaisesRegex(ValueError, "current review_sha256"):
            validate_annotations(rows, chunks, {})

    def test_source_quote_and_actual_context_are_checked(self):
        chunks, rows = fixtures()
        rows[0]["answer_claims"][0]["supports"] = [{"chunk_id": "c2", "quote": chunks[1]["text"]}]
        rows[0]["review_sha256"] = review_digest(rows[0])
        with self.assertRaisesRegex(ValueError, "absent from its actual context"):
            validate_annotations(rows, chunks, {})
        rows[0]["answer_claims"][0]["supports"][0]["quote"] = "fabricated quote"
        rows[0]["review_sha256"] = review_digest(rows[0])
        with self.assertRaisesRegex(ValueError, "quote must occur exactly"):
            validate_annotations(rows, chunks, {})

    def test_section_page_and_explicit_group_leakage_are_rejected(self):
        for mode in ("section", "page", "group", "fact"):
            chunks, rows = fixtures()
            if mode == "section":
                chunks[1]["section_path"] = chunks[0]["section_path"]
            elif mode == "page":
                chunks[1]["pdf_page"] = chunks[0]["pdf_page"]
            elif mode == "group":
                rows[1]["split_group"] = rows[0]["split_group"]
            else:
                rows[1]["required_facts"][0]["text"] = rows[0]["required_facts"][0]["text"]
            rows[1]["review_sha256"] = review_digest(rows[1])
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Split leakage"):
                validate_annotations(rows, chunks, {})

    def test_schedule_keeps_partial_accumulation_and_full_epochs(self):
        self.assertEqual(training_schedule(41, 1, 8, 1)["optimizer_steps"], 6)
        self.assertEqual(training_schedule(800, 1, 8, 1)["optimizer_steps"], 100)
        self.assertEqual(training_schedule(41, 2, 8, 2)["optimizer_steps"], 6)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            training_schedule(41, 1, 8, .762)

    def test_raw_jsonl_cannot_start_training(self):
        with self.assertRaisesRegex(ValueError, "prepare-sft artifact"):
            finetune(ROOT / "examples/finetune_train.jsonl", {}, "config.yaml", "unused")

    def test_attach_context_resets_approval_and_uses_packer(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            chunks, annotations = fixtures()
            retrieval = folder / "retrieval"
            m, _ = begin(retrieval, signature("retrieval", {}, {}, []))
            write_jsonl(retrieval / "retrieval.jsonl", [
                {"query_id": r["query_id"], "question": r["question"], "candidates": [c]}
                for r, c in zip(annotations, chunks)])
            finish(retrieval, m, ["retrieval.jsonl"])
            write_jsonl(folder / "drafts.jsonl", annotations)
            attach_context(folder / "drafts.jsonl", retrieval, generation_config(), "unused", folder / "packed", Backend())
            packed = read_jsonl(folder / "packed/drafts.jsonl")
            self.assertTrue(all(r["review_status"] == "pending" and r["review_sha256"] is None for r in packed))
            self.assertIn("[E1] PDF page 1", packed[0]["context"])


class EvaluationTests(unittest.TestCase):
    def test_ab_rejects_changed_prompt_base_model_or_context_annotations(self):
        for change in ("prompt", "base", "relevance"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                data = dataset(folder)
                for role in ("base", "candidate"):
                    cfg = generation_config()
                    if role == "candidate" and change == "base":
                        cfg["model_name"] = "different-base-model"
                    if role == "candidate" and change == "prompt":
                        (folder / "system.txt").write_text("A different instruction.")
                        cfg["system_prompt_file"] = str(folder / "system.txt")
                    pred = folder / role
                    evaluate(data, "dev", cfg, "unused", pred, _backend=Backend(),
                        _identity={"kind": "adapter", "training_id": "test", "checkpoint": "epoch-001"} if role == "candidate" else None)
                    review = reviews(read_jsonl(pred / "predictions.jsonl"))
                    if role == "candidate" and change == "relevance":
                        review[0]["context_relevant_spans"] = []
                    write_jsonl(folder / f"{role}.jsonl", review)
                    score_evaluation(pred, folder / f"{role}.jsonl", {}, folder / (role + "_scores"))
                with self.assertRaisesRegex(ValueError, "changed"):
                    paired_comparison(folder / "base_scores", folder / "candidate_scores", {}, "dev")

    def test_corrupt_partial_checkpoint_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = dataset(folder)
            out = folder / "pred"
            with patch("casml_b0.evaluation.write_jsonl", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    evaluate(data, "dev", generation_config(), "unused", out, _backend=Backend())
            checkpoint = next((out / "checkpoints").glob("*.json"))
            record = read_json(checkpoint)
            record["answer"] = "A damaged or externally changed partial record."
            write_json(checkpoint, record)
            with self.assertRaisesRegex(ValueError, "Invalid evaluation checkpoint"):
                evaluate(data, "dev", generation_config(), "unused", out, _backend=Backend())

    def test_interrupted_generation_resumes_verified_records(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = dataset(folder)
            out = folder / "predictions"
            backend = Backend()
            with patch("casml_b0.evaluation.write_jsonl", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    evaluate(data, "dev", generation_config(), "unused", out, _backend=backend)
            checkpoint = next((out / "checkpoints").glob("*.json"))
            saved = checkpoint.read_bytes()
            backend.generate = Mock(side_effect=AssertionError("Should reuse persisted prediction"))
            evaluate(data, "dev", generation_config(), "unused", out, _backend=backend)
            self.assertEqual(checkpoint.read_bytes(), saved)
            backend.generate.assert_not_called()
            (out / "predictions.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_artifact(out, "eval_predictions")

    def test_score_full_claim_coverage_and_coherent_reference_alternative(self):
        p = {"query_id": "q", "split_group": "g", "context": "first second third", "answer": "Correct. Wrong.",
             "required_facts": [{"fact_id": "f1"}, {"fact_id": "f2"}], "evidence_coverage": 1,
             "references": {"pages": [1], "sections": ["B"]}, "finish_reason": "eos",
             "gold_reference_sets": [{"pages": [1], "sections": ["A"]}, {"pages": [2], "sections": ["B"]}]}
        review = reviews([p])[0]
        review["claims"] = [{"text": "Correct.", "correct": True, "supported": True, "fact_ids": ["f1"]},
                            {"text": "Wrong.", "correct": False, "supported": False, "fact_ids": []}]
        review["context_relevant_spans"] = [[0, 5], [0, 12]]
        scored = score_row(p, review)
        self.assertAlmostEqual(scored["CP"], 2 / 3)
        self.assertEqual(scored["AF"], .5)
        self.assertEqual(scored["AC"], .5)
        self.assertEqual(scored["RA"], .5)  # Cannot combine pages of A with sections of B.
        review["claims"].pop()
        with self.assertRaisesRegex(ValueError, "full generated answer"):
            score_row(p, review)

    def test_complete_scoring_and_selection_workflow_requires_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = dataset(folder)
            candidate = {"kind": "adapter", "training_id": "test", "checkpoint": "epoch-001"}
            for split in ("dev", "holdout"):
                for role, good in (("base", False), ("candidate", True)):
                    pred = folder / f"{split}_{role}_pred"
                    evaluate(data, split, generation_config(), "unused", pred, _backend=Backend(),
                             _identity=candidate if role == "candidate" else None)
                    path = folder / f"{split}_{role}_reviews.jsonl"
                    write_jsonl(path, reviews(read_jsonl(pred / "predictions.jsonl"), good))
                    score_evaluation(pred, path, {}, folder / f"{split}_{role}_scores")
            cfg = {"min_dev_examples": 1, "min_holdout_examples": 1, "min_groups": 1, "bootstrap_samples": 100}
            base, cand = folder / "dev_base_scores", folder / "dev_candidate_scores"
            select_model(base, cand, cfg, folder / "dev_only")
            self.assertEqual(read_json(folder / "dev_only/report.json")["status"], "insufficient_evidence")
            select_model(base, cand, cfg, folder / "confirmed", folder / "holdout_base_scores", folder / "holdout_candidate_scores")
            report = read_json(folder / "confirmed/report.json")
            self.assertEqual(report["status"], "selected")
            self.assertFalse(report["official_metric_verified"])
            select_model(base, cand, {}, folder / "production", folder / "holdout_base_scores", folder / "holdout_candidate_scores")
            self.assertEqual(read_json(folder / "production/report.json")["status"], "insufficient_evidence")

    def test_review_cannot_be_reused_for_a_changed_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            evaluate(dataset(folder), "dev", generation_config(), "unused", folder / "pred", _backend=Backend())
            p = read_jsonl(folder / "pred/predictions.jsonl")[0]
            review = reviews([p])[0]
            p["answer"] = "Changed answer."
            with self.assertRaisesRegex(ValueError, "stale prediction"):
                score_row(p, review)

    def test_incomplete_export_cannot_be_selected_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            begin(directory, signature("finetune", {}, {}, []))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                selected_generation_config(directory)

    def test_completed_training_is_not_a_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            m, _ = begin(directory, signature("finetune", {}, {}, []))
            write_json(Path(directory) / "report.json", {"workflow": "reviewed-v2"})
            finish(directory, m, ["report.json"])
            with self.assertRaisesRegex(ValueError, "not a training candidate"):
                selected_generation_config(directory)


if __name__ == "__main__":
    unittest.main()
