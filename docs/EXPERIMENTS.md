# Cải thiện từng phần mà không sửa cả pipeline

## Đổi gì thì chạy lại gì?

| Thay đổi | Phần sửa | Chạy lại tối thiểu |
|---|---|---|
| Prompt, LLM, decoding, độ dài đáp án | prompts / generate.yaml / llm.py | generate → export |
| Chọn 4 thay vì 6 chunk, budget, bỏ trùng, format context | context.py / generate.yaml | generate → export |
| Tăng top 20 lên top 50, thêm hybrid/reranker | retrieve.yaml / retrieval.py | retrieve → generate → export; hybrid có thể cần index mới |
| Embedding model hoặc query prefix | index.yaml / embedding.py | index → retrieve → generate → export |
| Chunk size, làm sạch, OCR, ánh xạ mục/trang | prepare.yaml / prepare.py / page map | prepare → index → retrieve → generate → export |
| Quy ước trang CSV, kiểu số/chuỗi, hiển thị audit | export.yaml / exporting.py | export |
| Nội dung/tập câu hỏi | queries.json | retrieve → generate → export; giữ corpus/index |

Ngoại lệ có lý do: nếu cache chỉ chứa 20 ứng viên thì generation không thể tìm ứng viên thứ 21; nếu đổi embedding khiến chunk vượt giới hạn model mới thì phải chia lại chunk. B0 báo rõ thay vì âm thầm cắt dữ liệu.

## Thử generation độc lập

1. Chốt `artifacts/retrieval_v1` và giữ nguyên nó.
2. Sao chép `configs/generate.yaml` thành `configs/generate_g02.yaml`, đổi một yếu tố như prompt hoặc `context_top_k`.
3. Nếu thử prompt mới, sao chép prompt thành file mới và sửa đường dẫn trong YAML; đường dẫn được tính từ vị trí file YAML.
4. Chạy generation vào `runs/g02`; export vào `outputs/g02`.
5. So sánh đáp án trên cùng query IDs và cùng retrieval artifact ID. Đọc `evidence`, `input_tokens`, `output_tokens` và audit, không chỉ nhìn độ dài đáp án.

```bash
python -m casml_b0 generate --retrieval artifacts/retrieval_v1 --config configs/generate_g02.yaml --out runs/g02
python -m casml_b0 export --run runs/g02 --queries data/queries.json --config configs/export.yaml --pdf data/book.pdf --out outputs/g02
```

Thử đổi `context_top_k` là thí nghiệm context packing, không phải so sánh thuần model. Muốn so sánh model sạch hơn, giữ prompt/packing/decoding tương đương và kiểm tra evidence thực tế vì tokenizer khác có thể làm budget chọn khác số chunk.

## Kiểm tra chất lượng theo từng tầng

**Dữ liệu:** lấy mẫu trang có nhiều cột, bảng, đầu/cuối chương; kiểm tra text và số trang. Sửa ingestion nếu text sai trước khi tune retrieval.

**Retrieval:** tự lập một tập câu hỏi kiểm tra từ chính sách, ghi trang/đoạn nguồn đúng sau khi đọc thủ công. Đo tỷ lệ câu có nguồn đúng trong top-k (Recall@k/Hit@k theo định nghĩa đã chọn). B0 chưa cung cấp nhãn gold hoặc evaluator metric này. Không lấy đáp án sinh bởi model làm ground truth.

**Generation:** cố định retrieval cache; đánh giá đúng ý, đầy đủ và mọi khẳng định có được context hỗ trợ không. Nếu evidence đã có câu trả lời nhưng đáp án sai, xử lý generation. Nếu cache thiếu nguồn đúng, xử lý retrieval.

**Submission:** kiểm tra ID/schema/references và quy ước đánh số trang. Exporter dùng schema công bố trực tiếp; sample CSV chỉ là đối chiếu tùy chọn. Validation kỹ thuật có sẵn không thay thế metric chính thức.

## Lộ trình sau B0

1. Chốt một B0 chạy toàn bộ dữ liệu thật và lưu artifact/config/hash.
2. Cải thiện extraction/page map khi audit phát hiện sai.
3. R1 hiện tại: BGE-small + BM25 + weighted RRF + cross-encoder reranking.
4. Ablation retrieval: giữ generation cố định, lần lượt tắt BM25 hoặc reranker để đo Recall@k và điểm leaderboard.
5. Ablation generation: giữ cache hybrid cố định, so Qwen 0.5B/1.5B và context top 3/4/6; chỉ thay một yếu tố mỗi lượt.
6. Chỉ thêm HyDE nếu có kiểm tra riêng về lợi ích, độ trễ và chi phí. HyDE là query transformation trong nhánh retrieval, không đưa vào generation trả lời cuối.
