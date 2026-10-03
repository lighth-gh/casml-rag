import json
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from casml_b0.cli import main
from casml_b0.artifacts import begin, finish, signature, write_json
from casml_b0.finetuning import AnswerCollator, read_examples, split_examples, tokenize_example
from scripts.build_notebooks import FINETUNED_GENERATION, FINETUNING

ROOT = Path(__file__).resolve().parents[1]


class ChatTokenizer:
    """Small deterministic template exposing prompt/answer/EOS boundaries."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        result = []
        for message in messages:
            result += [1] + [ord(c) for c in message["content"]] + [2]
        return result + [1] if add_generation_prompt else result


class FineTuningTests(unittest.TestCase):
    def test_jsonl_requires_labels_and_unique_questions(self):
        original = read_examples(ROOT / "examples/finetune_train.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "train.jsonl"
            for rows, message in (([], "empty"), ([{"query_id": "1"}], "question"),
                                  ([{**original[0], "answer": " "}], "answer"),
                                  ([original[0], {**original[0], "query_id": "different"}], "Duplicate")):
                file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_examples(file)

    def test_split_is_deterministic_disjoint_and_does_not_mutate_input(self):
        rows = read_examples(ROOT / "examples/finetune_train.jsonl")
        before = list(rows)
        train, validation = split_examples(rows, seed=17)
        self.assertEqual((train, validation), split_examples(rows, seed=17))
        self.assertEqual(rows, before)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        self.assertFalse({row["query_id"] for row in train} & {row["query_id"] for row in validation})
        with self.assertRaisesRegex(ValueError, "at least two"):
            split_examples(rows[:1])

    def test_explicit_validation_rejects_id_and_normalized_question_leakage(self):
        rows = read_examples(ROOT / "examples/finetune_train.jsonl")
        for duplicate in (rows[0], {**rows[0], "query_id": "new", "question": rows[0]["question"].upper()}):
            with self.assertRaisesRegex(ValueError, "overlap"):
                split_examples(rows, [duplicate])
        self.assertEqual(split_examples(rows[:2], rows[2:]), (rows[:2], rows[2:]))

    def test_only_answer_and_end_of_turn_are_supervised(self):
        row = {"query_id": "1", "question": "Q", "context": "C", "answer": "A"}
        encoded = tokenize_example(row, ChatTokenizer(), "S", "{question} {context}", 100)
        self.assertEqual([x for x in encoded["labels"] if x != -100], [ord("A"), 2])
        self.assertEqual(len(encoded["labels"]), len(encoded["input_ids"]))
        with self.assertRaisesRegex(ValueError, "No silent truncation"):
            tokenize_example(row, ChatTokenizer(), "S", "{question} {context}", 3)
        broken = Mock()
        broken.apply_chat_template.side_effect = [[1, 2], [3, 4, 5]]
        with self.assertRaisesRegex(ValueError, "prefix"):
            tokenize_example(row, broken, "S", "{question} {context}", 100)

    def test_padding_is_masked_without_masking_real_eos(self):
        features = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                    {"input_ids": [2], "attention_mask": [1], "labels": [2]}]
        fake_torch = SimpleNamespace(long="long", tensor=lambda values, dtype: values)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            batch = AnswerCollator(2)(features)
        self.assertEqual(batch["input_ids"], [[1, 2], [2, 2]])
        self.assertEqual(batch["labels"], [[-100, 2], [2, -100]])
        self.assertEqual(batch["attention_mask"], [[1, 1], [1, 0]])

    def test_cli_routes_train_and_validation_without_importing_ml(self):
        with patch("casml_b0.finetuning.finetune", return_value={
                "stage": "finetune", "artifact_id": "123456789012", "complete": True}) as train:
            main(["finetune", "--train", "train.jsonl", "--validation", "val.jsonl",
                  "--config", str(ROOT / "configs/finetune.yaml"), "--out", "new-model"])
        self.assertEqual(train.call_args.args[0], "train.jsonl")
        self.assertEqual(train.call_args.args[-1], "val.jsonl")

    def test_notebook_can_explicitly_disable_training(self):
        stage = Mock()
        with patch("builtins.print"):
            exec(FINETUNING, {"WORK": ROOT, "stage": stage})
        stage.assert_not_called()

    def test_notebook_selects_moved_model_then_restores_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            artifact = work / "moved-model"
            artifact.mkdir()
            (work / "configs").mkdir()
            manifest, _ = begin(artifact, signature("finetune", {}, {}, []))
            (artifact / "generate.yaml").write_text(yaml.safe_dump({
                "model_name": "/old/model", "system_prompt_file": "system.txt", "user_prompt_file": "user.txt"}))
            write_json(artifact / "report.json", {"workflow": "selected-v2", "merge_verified": True, "reload_verified": True,
                       "selection_status": "selected", "training_id": "train", "checkpoint": "epoch-001"})
            write_json(artifact / "selection.json", {"status": "selected", "model": {
                       "kind": "adapter", "training_id": "train", "checkpoint": "epoch-001"}})
            finish(artifact, manifest, ["generate.yaml", "report.json", "selection.json"])
            namespace = {"WORK": work, "Path": Path, "yaml": yaml, "sys": sys, "ROOT": ROOT,
                         "FINETUNE_ENABLED": True, "FINETUNE_OUT": artifact,
                         "RUN_NAME": "baseline", "BASELINE_GEN_CONFIG": work / "base.yaml"}
            exec(FINETUNED_GENERATION.replace("FINETUNED_ARTIFACT = None", f"FINETUNED_ARTIFACT = {str(artifact)!r}"), namespace)
            config = yaml.safe_load(namespace["GEN_CONFIG"].read_text())
            self.assertEqual(config["model_name"], str(artifact.resolve() / "model"))
            self.assertEqual(config["system_prompt_file"], str(artifact.resolve() / "system.txt"))
            self.assertIn("ft_" + manifest["artifact_id"][:12], namespace["RUN"].name)
            namespace["FINETUNE_ENABLED"] = False
            exec(FINETUNED_GENERATION, namespace)
            self.assertEqual(namespace["GEN_CONFIG"], work / "base.yaml")
            self.assertEqual(namespace["RUN"], work / "runs/baseline")

    def test_notebook_cells_compile(self):
        notebook = json.loads((ROOT / "notebooks/CASML_R1_end_to_end.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")


if __name__ == "__main__":
    unittest.main()
