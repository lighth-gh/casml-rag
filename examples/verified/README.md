# Kết quả B0 đã chạy trên sách mẫu

Thư mục này ghi lại một lần chạy thật với **BGE-small + FAISS + Qwen2.5-0.5B-Instruct**, trên `examples/demo_book.pdf`. Không phải dữ liệu/kết quả CASML chính thức.

- `output/audit.html`: mở sau khi giải nén, bấm nguồn để mở `output/book.pdf` đúng trang.
- `output/submission.csv`: 3 dòng mẫu đúng schema nội bộ.
- `retrieval/`: cache tự chứa để thử generation; giữ nguyên manifest và JSONL.
- `generation/`: đáp án, prompt/evidence thực tế, checkpoint, token, thời gian.
- `verification.json`: thông tin kiểm tra và môi trường.

Từ thư mục project, sau khi cài dependency generation, có thể thử lại mà không dựng index:

```bash
python -m casml_b0 generate --retrieval examples/verified/retrieval --config configs/generate.yaml --out runs/from_saved_example
```

Q003 trong bản mẫu trả lời chưa đủ ý; giữ nguyên để làm mốc cải thiện generation. Xem `docs/VERIFICATION.md`.
