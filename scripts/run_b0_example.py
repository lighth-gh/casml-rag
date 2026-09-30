"""Run the actual B0 models on the included synthetic PDF (not contest scoring)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
steps = [
    ["prepare", "--pdf", "examples/demo_book.pdf", "--config", "configs/prepare.yaml",
     "--page-map-override", "examples/page_map_override.csv", "--out", "artifacts/example_corpus"],
    ["index", "--corpus", "artifacts/example_corpus", "--config", "configs/index.yaml",
     "--out", "artifacts/example_index"],
    ["retrieve", "--index", "artifacts/example_index", "--queries", "examples/queries.json",
     "--config", "configs/retrieve.yaml", "--out", "artifacts/example_retrieval"],
    ["generate", "--retrieval", "artifacts/example_retrieval", "--config", "configs/generate.yaml",
     "--out", "runs/example_b0"],
    ["export", "--run", "runs/example_b0", "--queries", "examples/queries.json",
     "--config", "configs/export.yaml", "--sample", "examples/sample_submission.csv",
     "--pdf", "examples/demo_book.pdf", "--out", "outputs/example_b0"],
]
for step in steps:
    subprocess.run([sys.executable, "-m", "casml_b0", *step], cwd=ROOT, check=True)
print("Open outputs/example_b0/audit.html and outputs/example_b0/submission.csv")
