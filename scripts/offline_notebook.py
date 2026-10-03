"""Offline inference cells, imported by build_notebooks.py."""

SETUP = '''from pathlib import Path
import os, subprocess, sys, shutil

# Attach source v2, a complete retrieval cache, queries, and a local model snapshot.
ROOT = Path("/kaggle/input/casml-v2-assets/casml_b0_starter")
RETRIEVAL = Path("/kaggle/input/casml-v2-assets/retrieval_hybrid_r2")
QUERIES = Path("/kaggle/input/casml-generative-ai-hackathon/queries.json")
BASE_MODEL = Path("/kaggle/input/casml-v2-assets/qwen2.5-1.5b-instruct")
FINETUNED_ARTIFACT = None  # Only a selected-v2 merge-model output; None means baseline.
WHEELHOUSE = None  # Optional local wheels matching the Kaggle Python/CUDA environment.
WORK = Path("/kaggle/working/casml_offline_v2")
WORK.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
if not (ROOT / "casml_b0/cli.py").is_file():
    raise FileNotFoundError("Attach source v2 and set ROOT. This notebook never clones/pulls a repository.")
if WHEELHOUSE is not None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(WHEELHOUSE),
                    "-r", str(ROOT / "requirements-generation.txt"), "-r", str(ROOT / "requirements-export.txt")], check=True)
import yaml
sys.path.insert(0, str(ROOT))
from casml_b0.artifacts import load_artifact, read_jsonl
from casml_b0.contracts import load_queries, validate_retrieval
load_artifact(RETRIEVAL, "retrieval")
queries = load_queries(QUERIES)
retrieved = read_jsonl(RETRIEVAL / "retrieval.jsonl")
validate_retrieval(retrieved)
if {r["query_id"]: r["question"] for r in queries} != {r["query_id"]: r["question"] for r in retrieved}:
    raise ValueError("Retrieval cache must match ALL query IDs/questions for this submission.")
if FINETUNED_ARTIFACT is None:
    if not (BASE_MODEL / "config.json").is_file():
        raise FileNotFoundError("Set BASE_MODEL to a complete local model/tokenizer snapshot.")
    selected_config = yaml.safe_load((ROOT / "configs/generate.yaml").read_text(encoding="utf-8"))
    selected_config.update(model_name=str(BASE_MODEL), local_files_only=True)
    for key in ("system_prompt_file", "user_prompt_file"):
        selected_config[key] = str((ROOT / "configs" / selected_config[key]).resolve())
    model_tag = "baseline"
else:
    from casml_b0.finetuning import selected_generation_config
    manifest, selected_config = selected_generation_config(FINETUNED_ARTIFACT)
    model_tag = manifest["artifact_id"][:12]
GEN_CONFIG = WORK / f"generate_{model_tag}.yaml"
GEN_CONFIG.write_text(yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8")
RUN, OUTPUT = WORK / f"run_{model_tag}", WORK / f"output_{model_tag}"

def stage(*args):
    subprocess.run([sys.executable, "-m", "casml_b0", *map(str, args)], cwd=ROOT, check=True)
print("Offline model:", selected_config["model_name"])
'''

RUN = '''stage("generate", "--retrieval", RETRIEVAL, "--config", GEN_CONFIG, "--out", RUN)
stage("export", "--run", RUN, "--queries", QUERIES, "--config", ROOT / "configs/export.yaml", "--out", OUTPUT)
shutil.copyfile(OUTPUT / "submission.csv", Path("/kaggle/working/submission.csv"))
print((OUTPUT / "validation.json").read_text(encoding="utf-8"))
print("Submission: /kaggle/working/submission.csv")
'''
