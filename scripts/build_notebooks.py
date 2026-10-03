"""Create portable Kaggle/local notebooks with ordinary Python cells only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(True),
            "execution_count": None, "outputs": []}


SETUP = '''from pathlib import Path
import subprocess, sys

# Notebook tự clone mã nguồn khi chưa có project. Bật Internet trên Kaggle.
REPO_URL = "https://github.com/lighth-gh/casml-rag.git"
REPO_REF = "main"
ROOT_OVERRIDE = None  # Hoặc đường dẫn tới một bản project đã có.
UPDATE_REPO = False   # Đổi thành True để git pull --ff-only khi chạy lại cell.

def is_project(path):
    return (path / "pyproject.toml").is_file() and (path / "casml_b0/cli.py").is_file()

if ROOT_OVERRIDE:
    ROOT = Path(ROOT_OVERRIDE).expanduser().resolve()
else:
    local = [Path.cwd(), *Path.cwd().parents]
    candidates = list(dict.fromkeys(p.resolve() for p in local if is_project(p)))
    if len(candidates) > 1:
        raise RuntimeError(f"Tìm thấy nhiều project: {candidates}. Hãy đặt ROOT_OVERRIDE.")
    if candidates:
        ROOT = candidates[0]
    else:
        base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
        ROOT = (base / "casml-rag").resolve()
        if ROOT.exists() and not is_project(ROOT):
            raise RuntimeError(f"{ROOT} đã tồn tại nhưng không phải project CASML; hãy đổi ROOT_OVERRIDE hoặc tên thư mục.")
        if not ROOT.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(ROOT)],
                check=True,
            )

if not is_project(ROOT):
    raise FileNotFoundError(f"Không tìm thấy project CASML tại {ROOT}")
if UPDATE_REPO:
    if not (ROOT / ".git").is_dir():
        raise RuntimeError("UPDATE_REPO chỉ dùng được với thư mục được git clone.")
    subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only", "origin", REPO_REF], check=True)

if Path("/kaggle/working").exists():
    # Artifact nằm ngoài repo để git pull không đụng vào kết quả đã chạy.
    WORK = Path("/kaggle/working/casml_b0_work")
else:
    WORK = ROOT
WORK.mkdir(parents=True, exist_ok=True)
print("Project:", ROOT)
print("Artifacts:", WORK)

def stage(*args):
    subprocess.run([sys.executable, "-m", "casml_b0", *map(str, args)], cwd=ROOT, check=True)
'''

PATHS = '''from pathlib import Path

# Có thể đặt đường dẫn chính xác; để None thì notebook tự dò trong INPUT_ROOT.
INPUT_ROOT_OVERRIDE = None
PDF_OVERRIDE = None
QUERIES_OVERRIDE = None
SAMPLE_OVERRIDE = None

AUTO_PDF = __AUTO_PDF__
REQUIRE_PDF = __REQUIRE_PDF__
AUTO_QUERIES = __AUTO_QUERIES__
AUTO_SAMPLE = __AUTO_SAMPLE__
REQUIRE_SAMPLE = __REQUIRE_SAMPLE__

import csv, json

INPUT_ROOT = (Path(INPUT_ROOT_OVERRIDE).expanduser().resolve() if INPUT_ROOT_OVERRIDE else
              (Path("/kaggle/input") if Path("/kaggle/input").exists() else ROOT / "data"))

def _explicit_path(value, label):
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = INPUT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}_OVERRIDE không tồn tại: {path}")
    return path

def _show(paths):
    paths = list(paths)
    shown = "\\n".join(f"  - {p}" for p in paths[:30])
    if len(paths) > 30:
        shown += f"\\n  ... và {len(paths) - 30} file khác"
    return shown or "  (không có)"

def _pick(label, override, candidates, preferred_names=(), required=True, inspected=()):
    explicit = _explicit_path(override, label)
    if explicit:
        return explicit
    candidates = sorted(dict.fromkeys(Path(p).resolve() for p in candidates))
    preferred = [p for p in candidates if p.name.casefold() in preferred_names]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    if not required:
        if len(candidates) > 1:
            print(f"Không tự chọn {label} vì có nhiều ứng viên; hãy đặt {label}_OVERRIDE nếu cần:\\n{_show(candidates)}")
        return None
    seen = candidates or list(inspected)
    reason = "không tìm thấy" if not candidates else "có nhiều ứng viên"
    raise FileNotFoundError(
        f"{label}: {reason} trong {INPUT_ROOT}. Hãy gắn Competition Data vào notebook "
        f"hoặc đặt {label}_OVERRIDE tới file chính xác.\\n{_show(seen)}"
    )

if any((AUTO_PDF, AUTO_QUERIES, AUTO_SAMPLE)) and not INPUT_ROOT.is_dir():
    raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {INPUT_ROOT}")

all_pdfs = sorted(INPUT_ROOT.rglob("*.pdf")) if AUTO_PDF else []
excluded_pdf_words = ("overview", "instruction", "rules", "guide", "readme")
book_pdfs = [p for p in all_pdfs if not any(word in p.name.casefold() for word in excluded_pdf_words)]
PDF = (_pick("PDF", PDF_OVERRIDE, book_pdfs,
             preferred_names=("book.pdf", "textbook.pdf", "psychology.pdf"),
             required=REQUIRE_PDF, inspected=all_pdfs)
       if AUTO_PDF or PDF_OVERRIDE else None)

def _looks_like_queries(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = value.get("queries") if isinstance(value, dict) else value
        return (isinstance(rows, list) and bool(rows) and
                all(isinstance(row, dict) and "query_id" in row and "question" in row for row in rows))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

all_json = sorted(INPUT_ROOT.rglob("*.json")) if AUTO_QUERIES else []
query_files = [p for p in all_json if _looks_like_queries(p)]
QUERIES = (_pick("QUERIES", QUERIES_OVERRIDE, query_files,
                 preferred_names=("queries.json",), required=True, inspected=all_json)
           if AUTO_QUERIES or QUERIES_OVERRIDE else None)

def _looks_like_sample(path):
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            fields = csv.DictReader(stream).fieldnames
        return fields is not None and set(fields) == {"ID", "context", "answer", "references"}
    except (OSError, UnicodeError, csv.Error):
        return False

all_csv = sorted(INPUT_ROOT.rglob("*.csv")) if AUTO_SAMPLE else []
sample_files = [p for p in all_csv if _looks_like_sample(p)]
SAMPLE = (_pick("SAMPLE", SAMPLE_OVERRIDE, sample_files,
                preferred_names=("sample_submission.csv",), required=REQUIRE_SAMPLE)
          if AUTO_SAMPLE or SAMPLE_OVERRIDE else None)

print("Input root:", INPUT_ROOT)
if AUTO_PDF or PDF_OVERRIDE:
    print("PDF:", PDF)
if AUTO_QUERIES or QUERIES_OVERRIDE:
    print("Queries:", QUERIES)
if AUTO_SAMPLE or SAMPLE_OVERRIDE:
    print("Sample:", SAMPLE)
    if SAMPLE is None:
        print("INFO: No sample_submission.csv was found. Export will use the documented "
              "ID, context, answer, references schema and query order.")

PAGE_MAP = None  # Đặt Path đến page_map_override.csv sau khi kiểm tra số trang/mục.
CORPUS = WORK / "artifacts/corpus_printed_v3"
INDEX = WORK / "artifacts/index_bge_v3"
RETRIEVAL = WORK / "artifacts/retrieval_hybrid_r2"
GEN_MAX_NEW_TOKENS = 384
GEN_RETRY_MAX_NEW_TOKENS = 384
# baseline: prompt bản 11 + top4; attribution: chỉ prompt mới.
# top3/top6: prompt bản 11, chỉ đổi số chunks. Dùng chung retrieval cache.
EXPERIMENT = "baseline"
EXPERIMENT_CONFIGS = {
    "baseline": ROOT / "configs/generate.yaml",
    "attribution": ROOT / "configs/experiments/generate_attribution.yaml",
    "top3": ROOT / "configs/experiments/generate_top3.yaml",
    "top6": ROOT / "configs/experiments/generate_top6.yaml",
}
if EXPERIMENT not in EXPERIMENT_CONFIGS:
    raise ValueError(f"EXPERIMENT phải là một trong {list(EXPERIMENT_CONFIGS)}")
RUN_NAME = f"hybrid_r4_{EXPERIMENT}_qwen15b_t{GEN_MAX_NEW_TOKENS}_report_only"
RUN = WORK / "runs" / RUN_NAME
OUTPUT = WORK / "outputs" / RUN_NAME
BASE_GEN_CONFIG = EXPERIMENT_CONFIGS[EXPERIMENT]
GEN_CONFIG = WORK / "configs" / f"generate_{EXPERIMENT}_t{GEN_MAX_NEW_TOKENS}_r{GEN_RETRY_MAX_NEW_TOKENS}.yaml"
BASE_EXPORT_CONFIG = ROOT / "configs/export.yaml"
EXPORT_CONFIG = WORK / "configs" / f"export_{EXPERIMENT}_documented_schema.yaml"
# Khi đổi prompt/model/config, dùng RUN và OUTPUT mới.
'''


def paths(auto_pdf=False, require_pdf=False, auto_queries=False, auto_sample=False,
          require_sample=False):
    values = {
        "__AUTO_PDF__": auto_pdf,
        "__REQUIRE_PDF__": require_pdf,
        "__AUTO_QUERIES__": auto_queries,
        "__AUTO_SAMPLE__": auto_sample,
        "__REQUIRE_SAMPLE__": require_sample,
    }
    source = PATHS
    for marker, value in values.items():
        source = source.replace(marker, repr(value))
    return code(source)


GENERATION_SETUP = '''# Tạo config runtime để notebook vẫn dùng giới hạn mới ngay cả khi repo clone còn config cũ.
import yaml

if ("retry_instruction" not in (ROOT / "casml_b0/generation.py").read_text(encoding="utf-8") or
        "write_diagnosis" not in (ROOT / "casml_b0/generation.py").read_text(encoding="utf-8") or
        "generate_grounded_answer" not in (ROOT / "casml_b0/generation.py").read_text(encoding="utf-8") or
        "grounding_report_enabled" not in (ROOT / "casml_b0/generation.py").read_text(encoding="utf-8") or
        not (ROOT / "casml_b0/validation.py").is_file() or
        "abort_after_consecutive_length_limited" not in (ROOT / "casml_b0/generation.py").read_text(encoding="utf-8") or
        "no_repeat_ngram_size" not in (ROOT / "casml_b0/llm.py").read_text(encoding="utf-8") or
        "eos_token_ids" not in (ROOT / "casml_b0/llm.py").read_text(encoding="utf-8") or
        "reciprocal_rank_fusion" not in (ROOT / "casml_b0/retrieval.py").read_text(encoding="utf-8")):
    raise RuntimeError("Mã nguồn đang dùng chưa có diagnosis, EOS guard, grounding validator hoặc fail-fast chống vòng lặp. Cập nhật repo chứa bản sửa "
                       "(UPDATE_REPO = True), hoặc đặt ROOT_OVERRIDE tới bản project mới rồi chạy lại setup.")
generation_config = yaml.safe_load(BASE_GEN_CONFIG.read_text(encoding="utf-8"))
generation_config["max_new_tokens"] = GEN_MAX_NEW_TOKENS
generation_config["retry_max_new_tokens"] = GEN_RETRY_MAX_NEW_TOKENS
generation_config.setdefault("eos_token_ids", [151645, 151643])
generation_config.setdefault("pad_token_id", 151643)
generation_config.setdefault("abort_after_consecutive_length_limited", 1)
generation_config.setdefault("retry_repetition_penalty", 1.15)
generation_config.setdefault("retry_no_repeat_ngram_size", 8)
generation_config["retry_instruction"] = ("Give a complete, concise answer in at most 180 words. "
                                          "State each relevant fact only once. Do not repeat sentences or continue "
                                          "a list unnecessarily. Finish the answer after addressing the question. "
                                          "Use only the supplied excerpts.")
generation_config["grounding_validator_enabled"] = False
generation_config["grounding_report_enabled"] = True
generation_config["grounding_validator_max_retries"] = 0
generation_config["grounding_retry_max_new_tokens"] = 224
generation_config["grounding_deterministic_repair"] = False
generation_config["grounding_repair_max_words"] = 140
generation_config["grounding_validator_retry_instruction"] = (
    "Use exact evidence only. Do not introduce any number, year, person, organization, place, "
    "named theory, or named work absent from the excerpts."
)
for key in ("system_prompt_file", "user_prompt_file"):
    prompt_path = Path(generation_config[key]).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = (BASE_GEN_CONFIG.parent / prompt_path).resolve()
    generation_config[key] = str(prompt_path)
GEN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
GEN_CONFIG.write_text(yaml.safe_dump(generation_config, sort_keys=False), encoding="utf-8")
BASELINE_GEN_CONFIG = GEN_CONFIG

export_config = yaml.safe_load(BASE_EXPORT_CONFIG.read_text(encoding="utf-8"))
export_config.pop("require_sample", None)
export_config["page_value_type"] = "integer"
export_config["fail_on_unsupported_claims"] = False
export_config["max_answer_words"] = 140
EXPORT_CONFIG.write_text(yaml.safe_dump(export_config, sort_keys=False), encoding="utf-8")
print("Experiment:", EXPERIMENT)
print("context_top_k:", generation_config["context_top_k"])
print("system_prompt_file:", generation_config["system_prompt_file"])
print("Grounding: report-only; no retry or sentence deletion")
print("Generation config:", GEN_CONFIG)
print("max_new_tokens:", generation_config["max_new_tokens"])
print("retry_max_new_tokens:", generation_config["retry_max_new_tokens"])
print("retry_repetition_penalty:", generation_config["retry_repetition_penalty"])
print("retry_no_repeat_ngram_size:", generation_config["retry_no_repeat_ngram_size"])
print("eos_token_ids:", generation_config["eos_token_ids"])
print("abort_after_consecutive_length_limited:", generation_config["abort_after_consecutive_length_limited"])
print("grounding_validator_max_retries:", generation_config["grounding_validator_max_retries"])
print("grounding_retry_max_new_tokens:", generation_config["grounding_retry_max_new_tokens"])
print("Export config:", EXPORT_CONFIG)
print("Run:", RUN)
'''

FINETUNING = '''# V2: fine-tune mặc định bật; cần artifact dữ liệu đã duyệt.
BUILD_DRAFTS = False
PREPARE_REVIEWED = False
FINETUNE_ENABLED = True  # Đặt False nếu chỉ tạo bản nháp hoặc muốn chạy baseline.
ANNOTATIONS_JSONL = None  # Bản sao JSONL do người đọc sách duyệt; xem docs/FINETUNING.md.
SFT_DRAFTS = WORK / "artifacts/book_drafts_v2"
REVIEWED_DATA = WORK / "artifacts/book_reviewed_v2"
FINETUNE_OUT = WORK / "artifacts/qwen15b_lora_candidates_v2"

if FINETUNE_ENABLED and not PREPARE_REVIEWED and not (REVIEWED_DATA / "manifest.json").is_file():
    raise FileNotFoundError(
        f"Fine-tune đang bật nhưng chưa có dữ liệu đã duyệt tại {REVIEWED_DATA}. "
        "Đặt ANNOTATIONS_JSONL và PREPARE_REVIEWED=True để tạo artifact reviewed_sft "
        "(tối thiểu 200 train, 80 dev, 80 holdout đã duyệt). "
        "Nếu chỉ tạo bản nháp ở 3b–3c hoặc muốn chạy baseline, đặt FINETUNE_ENABLED=False."
    )

if BUILD_DRAFTS or FINETUNE_ENABLED:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(ROOT / "requirements-finetune.txt")], check=True)
if BUILD_DRAFTS:
    draft_config = yaml.safe_load((ROOT / "configs/synthetic.yaml").read_text(encoding="utf-8"))
    draft_config["generation_config"] = str(BASELINE_GEN_CONFIG.resolve())
    draft_config_path = WORK / "configs/synthetic_runtime_v2.yaml"
    draft_config_path.write_text(yaml.safe_dump(draft_config, sort_keys=False), encoding="utf-8")
    stage("build-sft", "--corpus", CORPUS, "--config", draft_config_path, "--out", SFT_DRAFTS)
    print("Chỉ tạo bản nháp, chưa đủ điều kiện train:", SFT_DRAFTS / "drafts.jsonl")
if PREPARE_REVIEWED:
    if ANNOTATIONS_JSONL is None or not Path(ANNOTATIONS_JSONL).is_file():
        raise ValueError("Đặt ANNOTATIONS_JSONL đã duyệt trước khi prepare-sft.")
    stage("prepare-sft", "--corpus", CORPUS, "--annotations", ANNOTATIONS_JSONL,
          "--config", ROOT / "configs/reviewed.yaml", "--out", REVIEWED_DATA)
if FINETUNE_ENABLED:
    if not (REVIEWED_DATA / "manifest.json").is_file():
        raise FileNotFoundError("Cần artifact reviewed_sft: tối thiểu 200 train, 80 dev, 80 holdout đã duyệt.")
    fine_config = yaml.safe_load((ROOT / "configs/finetune.yaml").read_text(encoding="utf-8"))
    fine_config["generation_config"] = str(BASELINE_GEN_CONFIG.resolve())
    fine_config_path = WORK / "configs/finetune_runtime_v2.yaml"
    fine_config_path.write_text(yaml.safe_dump(fine_config, sort_keys=False), encoding="utf-8")
    stage("finetune", "--dataset", REVIEWED_DATA, "--config", fine_config_path, "--out", FINETUNE_OUT)
    print((FINETUNE_OUT / "report.json").read_text(encoding="utf-8"))
    print("Train xong chỉ tạo candidates. Duyệt dev_generations/base và từng checkpoint trước khi chọn.")
else:
    print("Fine-tune disabled: using baseline.")
'''

REVIEW_CONTEXT = '''# Tùy chọn: ghép context từ đúng retriever/packer dùng khi inference, TRƯỚC khi duyệt.
ATTACH_DRAFT_CONTEXT = False
DRAFT_RETRIEVAL = WORK / "artifacts/draft_retrieval_v2"
CONTEXT_DRAFTS = WORK / "artifacts/context_drafts_v2"
if ATTACH_DRAFT_CONTEXT:
    stage("retrieve", "--index", INDEX, "--queries", SFT_DRAFTS / "questions.json",
          "--config", ROOT / "configs/retrieve.yaml", "--out", DRAFT_RETRIEVAL)
    stage("attach-review-context", "--drafts", SFT_DRAFTS / "drafts.jsonl",
          "--retrieval", DRAFT_RETRIEVAL, "--config", BASELINE_GEN_CONFIG, "--out", CONTEXT_DRAFTS)
    print("Copy drafts.jsonl ra ngoài artifact, duyệt claims/sources/split, rồi quay lại PREPARE_REVIEWED:", CONTEXT_DRAFTS)
'''

MODEL_EVALUATION = '''# Chạy nhiều lần qua các giai đoạn review. Không dùng test queries làm nhãn.
# Điền đường dẫn bản review do người đọc sách chấm, KHÔNG sửa file trong artifact.
CANDIDATE_CHECKPOINT = None  # Ví dụ "epoch-001"; chọn theo dev, không theo train loss.
DEV_BASE_REVIEW = None
DEV_CANDIDATE_REVIEW = None
RUN_HOLDOUT = False  # Chỉ bật một lần sau khi khóa checkpoint theo dev.
HOLDOUT_BASE_REVIEW = None
HOLDOUT_CANDIDATE_REVIEW = None
MERGE_SELECTED = False
EVAL_WORK = WORK / "evaluation_v2"
MERGED_OUT = WORK / "artifacts/qwen15b_selected_v2"
if CANDIDATE_CHECKPOINT is not None:
    EVAL_WORK = EVAL_WORK / CANDIDATE_CHECKPOINT
    EVAL_WORK.mkdir(parents=True, exist_ok=True)
    dev_base = FINETUNE_OUT / "dev_generations/base"
    dev_candidate = FINETUNE_OUT / "dev_generations" / CANDIDATE_CHECKPOINT
    for name, predictions, reviews in (("dev_base", dev_base, DEV_BASE_REVIEW),
                                       ("dev_candidate", dev_candidate, DEV_CANDIDATE_REVIEW)):
        if reviews is not None:
            stage("score-eval", "--predictions", predictions, "--reviews", reviews,
                  "--config", ROOT / "configs/evaluation.yaml", "--out", EVAL_WORK / name)
    if RUN_HOLDOUT:
        for name, extra in (("holdout_base", []), ("holdout_candidate", ["--training", FINETUNE_OUT,
                                                                               "--checkpoint", CANDIDATE_CHECKPOINT])):
            stage("evaluate", "--dataset", REVIEWED_DATA, "--split", "holdout",
                  "--config", BASELINE_GEN_CONFIG, "--out", EVAL_WORK / (name + "_predictions"), *extra)
        for name, reviews in (("holdout_base", HOLDOUT_BASE_REVIEW), ("holdout_candidate", HOLDOUT_CANDIDATE_REVIEW)):
            if reviews is not None:
                stage("score-eval", "--predictions", EVAL_WORK / (name + "_predictions"), "--reviews", reviews,
                      "--config", ROOT / "configs/evaluation.yaml", "--out", EVAL_WORK / name)
    if DEV_BASE_REVIEW is not None and DEV_CANDIDATE_REVIEW is not None:
        confirm = HOLDOUT_BASE_REVIEW is not None and HOLDOUT_CANDIDATE_REVIEW is not None
        SELECTION = EVAL_WORK / ("selection_confirmed" if confirm else "selection_dev_only")
        args = ["select-model", "--baseline", EVAL_WORK / "dev_base", "--candidate", EVAL_WORK / "dev_candidate",
                "--config", ROOT / "configs/evaluation.yaml", "--out", SELECTION]
        if confirm:
            args += ["--holdout-baseline", EVAL_WORK / "holdout_base", "--holdout-candidate", EVAL_WORK / "holdout_candidate"]
        stage(*args)
        print((SELECTION / "report.json").read_text(encoding="utf-8"))
        if MERGE_SELECTED:
            stage("merge-model", "--training", FINETUNE_OUT, "--checkpoint", CANDIDATE_CHECKPOINT,
                  "--selection", SELECTION, "--config", ROOT / "configs/evaluation.yaml", "--out", MERGED_OUT)
else:
    print("Chưa chọn checkpoint. Hoàn tất review dev/holdout và merge ở 3d trước khi inference fine-tuned.")
'''

FINETUNED_GENERATION = '''# Chỉ trỏ tới output merge-model đã qua dev + holdout; không tự chọn sau train.
FINETUNED_ARTIFACT = WORK / "artifacts/qwen15b_selected_v2" if FINETUNE_ENABLED else None
if FINETUNED_ARTIFACT is not None:
    if not (Path(FINETUNED_ARTIFACT) / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Chưa có model fine-tuned đã chọn tại {FINETUNED_ARTIFACT}. "
            "Hoàn tất đánh giá dev/holdout và MERGE_SELECTED=True ở section 3d, "
            "hoặc đặt FINETUNED_ARTIFACT tới artifact merge-model đã hoàn tất. "
            "Để chạy baseline, đặt FINETUNE_ENABLED=False ở 3b rồi chạy lại cell chọn model."
        )
    sys.path.insert(0, str(ROOT))
    from casml_b0.finetuning import selected_generation_config
    fine_manifest, selected_config = selected_generation_config(FINETUNED_ARTIFACT)
    model_tag = fine_manifest["artifact_id"][:12]
    GEN_CONFIG = WORK / "configs" / f"generate_finetuned_{model_tag}.yaml"
    GEN_CONFIG.write_text(yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8")
    RUN = WORK / "runs" / f"{RUN_NAME}_ft_{model_tag}"
    OUTPUT = WORK / "outputs" / f"{RUN_NAME}_ft_{model_tag}"
    print("Using selected, merge-verified model:", selected_config["model_name"])
else:
    GEN_CONFIG = BASELINE_GEN_CONFIG
    RUN = WORK / "runs" / RUN_NAME
    OUTPUT = WORK / "outputs" / RUN_NAME
    print("Dùng model baseline:", GEN_CONFIG)
print("Generation config:", GEN_CONFIG)
print("Run:", RUN)
'''

STEPS = {
1: ('prepare', 'requirements-prepare.txt', '''args = ["prepare", "--pdf", PDF, "--config", ROOT / "configs/prepare.yaml", "--out", CORPUS]
if PAGE_MAP is not None:
    args += ["--page-map-override", PAGE_MAP]
stage(*args)
print((CORPUS / "report.json").read_text())
'''),
2: ('index', 'requirements-retrieval.txt', '''stage("index", "--corpus", CORPUS, "--config", ROOT / "configs/index.yaml", "--out", INDEX)
'''),
3: ('retrieve', 'requirements-retrieval.txt', '''stage("retrieve", "--index", INDEX, "--queries", QUERIES, "--config", ROOT / "configs/retrieve.yaml", "--out", RETRIEVAL)
print("Lưu cả thư mục này để thử generation ở session khác:", RETRIEVAL)
'''),
4: ('generate', 'requirements-generation.txt', '''# Nếu cache nằm ở Input, sửa RETRIEVAL tại đây, ví dụ:
# RETRIEVAL = Path("/kaggle/input/your-retrieval-cache/retrieval_hybrid_r1")
# Chỉ cần manifest.json + retrieval.jsonl. Không cần PDF/index/embedding.
try:
    stage("generate", "--retrieval", RETRIEVAL, "--config", GEN_CONFIG, "--out", RUN)
finally:
    diagnosis_path = RUN / "diagnosis.json"
    if diagnosis_path.is_file():
        print("Diagnosis:", diagnosis_path)
        from IPython.display import FileLink, display
        display(FileLink(str(diagnosis_path)))
print((RUN / "report.json").read_text())
'''),
5: ('export', 'requirements-export.txt', '''args = ["export", "--run", RUN, "--queries", QUERIES, "--config", EXPORT_CONFIG, "--out", OUTPUT]
if SAMPLE is not None:
    args += ["--sample", SAMPLE]
if PDF is not None:
    args += ["--pdf", PDF]  # Đặt PDF = None nếu chỉ có run generation.
stage(*args)
print((OUTPUT / "validation.json").read_text())
from IPython.display import FileLink, display
display(FileLink(str(OUTPUT / "submission.csv")))
display(FileLink(str(OUTPUT / "audit.html")))
'''),
}


def install(requirements):
    return code(f'subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "{requirements}")], check=True)\n')


def save(name, cells):
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:03d}"
    dest = ROOT / "notebooks" / name
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    intro = ('# CASML Hybrid R1 — end-to-end PDF → CSV\n\n'
             'Notebook phát triển cho toàn bộ câu hỏi: BGE-small + BM25 + reciprocal-rank fusion + '
             'cross-encoder reranking + Qwen2.5-1.5B-Instruct. Các stage vẫn nằm ở cell/section riêng '
             'để có thể chỉnh cấu hình và chạy lại từ đúng điểm cần thiết.\n\n'
             'Bật Internet để clone GitHub và tải model lần đầu; bật GPU nếu có. Sửa đường dẫn dữ liệu '
             'ở section 0. Notebook không tự nộp submission. Mỗi stage chạy trong process riêng để giải '
             'phóng bộ nhớ sau khi hoàn tất. Notebook luôn export theo schema '
             '`ID,context,answer,references`; sample_submission.csv chỉ là đầu vào tùy chọn. '
             'Trước khi nộp, kiểm tra page mapping trong validation/audit.')
    setup_note = ('## 0. Setup + input paths + runtime config\n\n'
                  'Chỉnh `ROOT_OVERRIDE`, `INPUT_ROOT_OVERRIDE`, `PDF_OVERRIDE`, `QUERIES_OVERRIDE` và '
                  '`SAMPLE_OVERRIDE` tại đây. Chọn `EXPERIMENT`: `baseline`, `attribution`, `top3`, '
                  'hoặc `top6`. Baseline khôi phục policy bản 11; các biến thể chỉ đổi một yếu tố. '
                  'Grounding chỉ báo cáo, không sửa đáp án hoặc chặn export. Các section phía dưới dùng chung đường dẫn này.')
    full = [md(intro), md(setup_note), code(SETUP),
            paths(auto_pdf=True, require_pdf=True, auto_queries=True, auto_sample=True,
                  require_sample=False),
            install("requirements.txt"), code(GENERATION_SETUP)]
    guidance = {
        1: ('PDF → chunks + printed-page map',
            'Chỉnh `configs/prepare.yaml`. Nếu đổi bước này, chạy lại tất cả section phía dưới.'),
        2: ('Dense index',
            'Chỉnh `configs/index.yaml`. Nếu đổi embedding/index, chạy lại từ section này.'),
        3: ('Hybrid retrieval + reranker',
            'Chỉnh `configs/retrieve.yaml` để tune dense/BM25/RRF/reranker, rồi chạy lại section 3–5.'),
        4: ('Qwen 1.5B generation',
            'Đổi `EXPERIMENT` ở section 0, chạy lại runtime config rồi section 4–5. '
            'Giữ nguyên retrieval; mỗi lựa chọn có run/output/config riêng.'),
        5: ('Validate + export submission',
            'Luôn tạo `submission.csv`; sample_submission.csv là tùy chọn. Kiểm tra `validation.json` và `audit.html` trước khi nộp.'),
    }
    for number, (name, requirements, command) in STEPS.items():
        heading, note = guidance[number]
        title = (f"## {number}. {heading}\n\n{note}\n\n"
                 'Đầu ra có manifest và checksum; cấu hình mới nên dùng thư mục output mới để tránh cache cũ.')
        if number == 4:
            full += [md("## 3b. Fine-tune v2: dữ liệu đã duyệt, mặc định bật\n\n"
                        "Đọc docs/FINETUNING.md. Bật BUILD_DRAFTS để tạo bản nháp; ghép context ở 3c, "
                        "duyệt rồi quay lại PREPARE_REVIEWED. FINETUNE_ENABLED chỉ dùng artifact đã duyệt. "
                        "LoRA r8, LR 1e-5, 1 epoch. Section 4 mặc định dùng model đã chọn và merge ở 3d; "
                        "thiếu dữ liệu/model thì dừng. Đặt FINETUNE_ENABLED=False khi chỉ tạo bản nháp "
                        "hoặc muốn chạy baseline."), code(FINETUNING),
                     md("## 3c. Context cho bản nháp trước khi duyệt\n\n"
                        "Chạy một lần sau BUILD_DRAFTS. Kiểm tra nhóm nguồn, facts thiếu và số trang; "
                        "không tự xem nhãn tổng hợp là gold."), code(REVIEW_CONTEXT),
                     md("## 3d. Đánh giá và chọn checkpoint\n\n"
                        "Chấm từng claim độc lập trên dev; khóa checkpoint rồi mới mở holdout. "
                        "S_proxy = 0.2 CP + 0.2 AF + 0.4 AC + 0.2 RA là proxy nội bộ, không phải scorer Kaggle. "
                        "Loss giảm không đủ để chọn model. Review JSONL theo docs/FINETUNING.md."), code(MODEL_EVALUATION)]
        full += [md(title)]
        if number == 4:
            full += [code(FINETUNED_GENERATION)]
        full += [code(command)]
    save("CASML_R1_end_to_end.ipynb", full)
    try:
        from .offline_notebook import SETUP as offline_setup, RUN as offline_run
    except ImportError:
        from offline_notebook import SETUP as offline_setup, RUN as offline_run
    save("CASML_R1_inference_offline.ipynb", [
        md("# CASML v2 — inference offline\n\n"
           "Bật GPU, tắt Internet. Gắn code v2, model snapshot/selected artifact, retrieval cache đầy đủ "
           "và queries.json. Nếu cần cài dependencies, chuẩn bị wheelhouse đúng Python/CUDA từ trước. "
           "Notebook không tạo nhãn hoặc train; không tự nộp. Baseline là mặc định. "
           "Cache retrieval phải khớp chính xác ID/question. Kiểm tra audit trước khi nộp."),
        code(offline_setup), code(offline_run)])
    print("Created development and offline inference notebooks")


if __name__ == "__main__":
    main()
