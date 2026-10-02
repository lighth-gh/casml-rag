import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from casml_b0.artifacts import begin, digest, finish, load_artifact, read_jsonl, signature, write_jsonl
from casml_b0.finetuning import read_examples, split_examples
from casml_b0.synthetic import build_sft, page_key, parse_pair, select_chunks
from scripts.build_notebooks import FINETUNING

ROOT = Path(__file__).resolve().parents[1]


def chunk(page):
    text = (f"Concept {page} describes how people process information during a specific psychological task. "
            "Researchers use careful observations to study this process in controlled experiments.")
    return {"chunk_id": f"c{page}", "doc_id": "book", "source_pdf": "book.pdf", "pdf_page": page,
            "printed_page": str(page), "section_path": ["Memory"], "section_method": "test",
            "text": text, "text_sha256": digest(text), "char_start": 0, "char_end": len(text)}


class Teacher:
    context_window = 4096

    def count_messages(self, messages):
        return 300

    def generate(self, payload, config):
        text = payload["messages"][-1]["content"].removeprefix("Excerpt:\n")
        concept = text.split()[1]
        return {"answer": json.dumps({"question": f"How does concept {concept} describe information processing?",
                                       "answer": text.split(". ")[0] + "."}), "finish_reason": "eos"}


class SyntheticTests(unittest.TestCase):
    def config(self):
        return {"generation_config": str(ROOT / "configs/generate.yaml"), "min_chunk_words": 1,
                "max_candidates": 10, "max_examples": 4, "min_examples": 2, "validation_fraction": 0.5}

    def corpus(self, directory):
        out = directory / "corpus"
        manifest, _ = begin(out, signature("corpus", {}, {}, []))
        write_jsonl(out / "chunks.jsonl", [chunk(n) for n in range(1, 9)])
        finish(out, manifest, ["chunks.jsonl"])
        return out

    def test_split_keeps_all_overlapping_chunks_on_same_page(self):
        chunks = [chunk(n) for n in range(1, 9)]
        extra = {**chunks[0], "chunk_id": "overlap", "text": chunks[0]["text"] + " Extra sentence."}
        extra.update(text_sha256=digest(extra["text"]), char_end=len(extra["text"]))
        chunks.append(extra)
        split = select_chunks(chunks, self.config())
        self.assertEqual(split, select_chunks(chunks, self.config()))
        self.assertFalse({page_key(c) for c in split["train"]} & {page_key(c) for c in split["validation"]})
        self.assertEqual(sum(map(len, split.values())), 9)

    def test_parser_rejects_invented_answer_and_keeps_source(self):
        source = chunk(1)
        pair = {"question": "How does concept one describe information processing?", "answer": source["text"]}
        accepted = parse_pair(json.dumps(pair), source, self.config())
        self.assertEqual(accepted["source"]["pdf_page"], 1)
        self.assertIn("[E1] PDF page 1", accepted["context"])
        self.assertEqual(accepted["label_source"], "synthetic_question_exact_source_answer")
        pair["answer"] += " This invented claim is absent."
        with self.assertRaisesRegex(ValueError, "exact_source_span"):
            parse_pair(json.dumps(pair), source, self.config())
        with self.assertRaisesRegex(ValueError, "invalid_json"):
            parse_pair("not JSON", source, self.config())

    def test_build_outputs_readable_training_data_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            corpus = self.corpus(folder)
            out = folder / "sft"
            with patch("casml_b0.synthetic.make_generator", return_value=Teacher()) as factory:
                manifest = build_sft(corpus, self.config(), folder / "config.yaml", out)
                self.assertTrue(manifest["complete"])
                self.assertEqual(factory.call_count, 1)
            train, val = read_examples(out / "train.jsonl"), read_examples(out / "validation.jsonl")
            self.assertEqual((len(train), len(val)), (2, 2))
            split_examples(train, val)
            with patch("casml_b0.synthetic.make_generator", side_effect=AssertionError("No reload")):
                self.assertEqual(build_sft(corpus, self.config(), folder / "config.yaml", out), manifest)
            (out / "train.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_artifact(out, "sft_data")

    def test_rejected_outputs_do_not_mark_dataset_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            corpus = self.corpus(folder)
            teacher = Teacher()
            teacher.generate = Mock(return_value={"answer": "{}", "finish_reason": "length"})
            with patch("casml_b0.synthetic.make_generator", return_value=teacher):
                with self.assertRaisesRegex(ValueError, "Training has NOT started"):
                    build_sft(corpus, self.config(), folder / "config.yaml", folder / "sft")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_artifact(folder / "sft", "sft_data")

    def test_interrupted_build_resumes_completed_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            corpus = self.corpus(folder)
            teacher = Teacher()
            original = teacher.generate
            teacher.generate = Mock(side_effect=[original({"messages": [{"content": "Excerpt:\n" + chunk(8)["text"]}]}, {}),
                                                 RuntimeError("interrupted")])
            with patch("casml_b0.synthetic.make_generator", return_value=teacher):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    build_sft(corpus, self.config(), folder / "config.yaml", folder / "sft")
            # Completed calls are persisted even if filtering rejects their content.
            checkpoints = list((folder / "sft/checkpoints").glob("*.json"))
            self.assertEqual(len(checkpoints), 1)
            saved = checkpoints[0].read_bytes()
            with patch("casml_b0.synthetic.make_generator", return_value=Teacher()):
                build_sft(corpus, self.config(), folder / "config.yaml", folder / "sft")
            self.assertEqual(checkpoints[0].read_bytes(), saved)

    def test_notebook_builds_jsonl_then_trains_without_test_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "configs").mkdir()
            corpus = self.corpus(work)
            calls = []

            def stage(*args):
                calls.append(args)
                out = Path(args[args.index("--out") + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / "report.json").write_text("{}")
                if args[0] == "build-sft":
                    (out / "train.jsonl").write_text("{}")
                    (out / "validation.jsonl").write_text("{}")

            with patch("builtins.print"):
                exec(FINETUNING, {"ROOT": ROOT, "WORK": work, "CORPUS": corpus, "Path": Path,
                                  "yaml": yaml, "subprocess": Mock(), "sys": Mock(), "stage": stage,
                                  "BASELINE_GEN_CONFIG": ROOT / "configs/generate.yaml"})
            self.assertEqual([args[0] for args in calls], ["build-sft", "finetune"])
            self.assertIn("--validation", calls[1])
            self.assertFalse(any("--queries" in args for args in calls))


if __name__ == "__main__":
    unittest.main()
