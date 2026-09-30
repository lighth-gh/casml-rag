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
                preferred_names=("sample_submission.csv",), required=False)
          if AUTO_SAMPLE or SAMPLE_OVERRIDE else None)

print("Input root:", INPUT_ROOT)
if AUTO_PDF or PDF_OVERRIDE:
    print("PDF:", PDF)
if AUTO_QUERIES or QUERIES_OVERRIDE:
    print("Queries:", QUERIES)
if AUTO_SAMPLE or SAMPLE_OVERRIDE:
    print("Sample:", SAMPLE)

PAGE_MAP = None  # Đặt Path đến page_map_override.csv sau khi kiểm tra số trang/mục.
CORPUS = WORK / "artifacts/corpus_v1"
INDEX = WORK / "artifacts/index_v1"
RETRIEVAL = WORK / "artifacts/retrieval_v1"
GEN_MAX_NEW_TOKENS = 512
RUN_NAME = f"b0_g01_t{GEN_MAX_NEW_TOKENS}"
RUN = WORK / "runs" / RUN_NAME
OUTPUT = WORK / "outputs" / RUN_NAME
BASE_GEN_CONFIG = ROOT / "configs/generate.yaml"
GEN_CONFIG = WORK / "configs" / f"generate_t{GEN_MAX_NEW_TOKENS}.yaml"
# Khi đổi prompt/model/config, dùng RUN và OUTPUT mới.
'''


def paths(auto_pdf=False, require_pdf=False, auto_queries=False, auto_sample=False):
    values = {
        "__AUTO_PDF__": auto_pdf,
        "__REQUIRE_PDF__": require_pdf,
        "__AUTO_QUERIES__": auto_queries,
        "__AUTO_SAMPLE__": auto_sample,
    }
    source = PATHS
    for marker, value in values.items():
        source = source.replace(marker, repr(value))
    return code(source)


GENERATION_SETUP = '''# Tạo config runtime để notebook vẫn dùng giới hạn mới ngay cả khi repo clone còn config cũ.
import yaml

generation_config = yaml.safe_load(BASE_GEN_CONFIG.read_text(encoding="utf-8"))
generation_config["max_new_tokens"] = GEN_MAX_NEW_TOKENS
for key in ("system_prompt_file", "user_prompt_file"):
    prompt_path = Path(generation_config[key]).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = (BASE_GEN_CONFIG.parent / prompt_path).resolve()
    generation_config[key] = str(prompt_path)
GEN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
GEN_CONFIG.write_text(yaml.safe_dump(generation_config, sort_keys=False), encoding="utf-8")
print("Generation config:", GEN_CONFIG)
print("max_new_tokens:", generation_config["max_new_tokens"])
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
# RETRIEVAL = Path("/kaggle/input/your-retrieval-cache/retrieval_v1")
# Chỉ cần manifest.json + retrieval.jsonl. Không cần PDF/index/embedding.
stage("generate", "--retrieval", RETRIEVAL, "--config", GEN_CONFIG, "--out", RUN)
print((RUN / "report.json").read_text())
'''),
5: ('export', 'requirements-export.txt', '''args = ["export", "--run", RUN, "--queries", QUERIES, "--config", ROOT / "configs/export.yaml", "--out", OUTPUT]
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
    intro = ('# CASML B0 — PDF → CSV\n\n'
             'BGE-small + FAISS + Qwen2.5-0.5B-Instruct. BM25 và Qwen 1.5B để nâng cấp sau. Đọc README trước khi chạy. '
             'Bật Internet để clone GitHub và tải model lần đầu; bật GPU nếu có. Sửa đường dẫn dữ liệu ở cell cấu hình. '
             'Notebook không tự nộp submission.\n\n'
             'Mỗi bước chạy process riêng. Đổi generation thì dùng notebook 04 trên cache có sẵn. '
             'Kiểm tra schema chính thức và số trang trước khi nộp.')
    full = [md(intro), code(SETUP), paths(auto_pdf=True, require_pdf=True, auto_queries=True, auto_sample=True),
            install("requirements.txt"), code(GENERATION_SETUP)]
    for number, (name, requirements, command) in STEPS.items():
        title = f"## {number}. {name}\n\nĐầu ra có manifest và checksum. Cấu hình mới cần thư mục output mới."
        full += [md(title), code(command)]
        stage_intro = f"# CASML B0 — {name}\n\nChỉ chạy bước {name}. Chỉnh đường dẫn artifact đã có trong cell cấu hình."
        if number == 4:
            stage_intro += "\n\nGeneration chỉ đọc cache retrieval; không import FAISS/SentenceTransformer, không cần corpus/index."
        if number == 5:
            stage_intro += "\n\nKhông tải LLM. Có thể đổi page_mode và xuất lại từ run generation cũ."
        path_options = {
            1: dict(auto_pdf=True, require_pdf=True),
            2: {},
            3: dict(auto_queries=True),
            4: {},
            5: dict(auto_pdf=True, auto_queries=True, auto_sample=True),
        }[number]
        cells = [md(stage_intro), code(SETUP), paths(**path_options), install(requirements)]
        if number == 4:
            cells.append(code(GENERATION_SETUP))
        cells.append(code(command))
        save(f"{number:02d}_{name}.ipynb", cells)
    save("00_run_b0.ipynb", full)
    print("Created 6 notebooks")


if __name__ == "__main__":
    main()
