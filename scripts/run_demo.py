"""Run all stages with explicit offline DEMO backends (no model downloads)."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
commands = [
    ["prepare", "--pdf", "examples/demo_book.pdf", "--page-map-override", "examples/page_map_override.csv", "--config", "configs/demo_prepare.yaml", "--out", "artifacts/demo_corpus"],
    ["index", "--corpus", "artifacts/demo_corpus", "--config", "configs/demo_index.yaml", "--out", "artifacts/demo_index"],
    ["retrieve", "--index", "artifacts/demo_index", "--queries", "examples/queries.json", "--config", "configs/retrieve.yaml", "--out", "artifacts/demo_retrieval"],
    ["generate", "--retrieval", "artifacts/demo_retrieval", "--config", "configs/demo_generate.yaml", "--out", "runs/demo"],
    ["export", "--run", "runs/demo", "--queries", "examples/queries.json", "--sample", "examples/sample_submission.csv", "--pdf", "examples/demo_book.pdf", "--config", "configs/demo_export.yaml", "--out", "outputs/demo"],
]
for command in commands:
    subprocess.run([sys.executable, "-m", "casml_b0", *command], cwd=root, check=True)
print("Open outputs/demo/audit.html and outputs/demo/submission.csv")
