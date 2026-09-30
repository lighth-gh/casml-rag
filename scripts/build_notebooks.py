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

PATHS = '''# SỬA đường dẫn cho đúng dữ liệu của bạn. Không dùng PDF overview làm sách.
DATA = Path("/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)") if Path("/kaggle/input").exists() else ROOT / "data"
PDF = DATA / "book.pdf"
QUERIES = DATA / "queries.json"
SAMPLE = None  # Đặt Path đến sample_submission.csv chính thức nếu có.
PAGE_MAP = None  # Đặt Path đến page_map_override.csv sau khi kiểm tra số trang/mục.
CORPUS = WORK / "artifacts/corpus_v1"
INDEX = WORK / "artifacts/index_v1"
RETRIEVAL = WORK / "artifacts/retrieval_v1"
RUN = WORK / "runs/b0_g01"
OUTPUT = WORK / "outputs/b0_g01"
GEN_CONFIG = ROOT / "configs/generate.yaml"
# Khi đổi prompt/model/config, dùng RUN và OUTPUT mới.
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
    full = [md(intro), code(SETUP), install("requirements.txt"), code(PATHS)]
    for number, (name, requirements, command) in STEPS.items():
        title = f"## {number}. {name}\n\nĐầu ra có manifest và checksum. Cấu hình mới cần thư mục output mới."
        full += [md(title), code(command)]
        stage_intro = f"# CASML B0 — {name}\n\nChỉ chạy bước {name}. Chỉnh đường dẫn artifact đã có trong cell cấu hình."
        if number == 4:
            stage_intro += "\n\nGeneration chỉ đọc cache retrieval; không import FAISS/SentenceTransformer, không cần corpus/index."
        if number == 5:
            stage_intro += "\n\nKhông tải LLM. Có thể đổi page_mode và xuất lại từ run generation cũ."
        save(f"{number:02d}_{name}.ipynb", [md(stage_intro), code(SETUP), install(requirements), code(PATHS), code(command)])
    save("00_run_b0.ipynb", full)
    print("Created 6 notebooks")


if __name__ == "__main__":
    main()
