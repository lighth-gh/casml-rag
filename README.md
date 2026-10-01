# CASML B0 — từ sách PDF đến CSV, từng bước độc lập

Pipeline R1 có thể chạy từ đầu, dùng **BGE-small-en-v1.5 + BM25 → reciprocal-rank fusion → cross-encoder reranking → Qwen2.5-1.5B-Instruct**. BM25 và dense được hợp nhất theo thứ hạng, không cộng trực tiếp hai loại điểm khác thang đo.

**Generation chỉ đọc cache retrieval chứa sẵn câu hỏi, đoạn văn và nguồn.** Đổi prompt, LLM, cách đóng gói context hoặc độ dài đáp án không cần sách PDF, index FAISS hay model embedding trong phiên generation. Có một kiểm tra tự động xóa corpus/index và chặn import retrieval để xác nhận ranh giới này.

Đây là pipeline tham khảo notebook CASML công khai, không phải bản tái lập nguyên trạng giải nhất/nhì. PDF trong `examples/` là tài liệu tổng hợp ngắn để kiểm tra kỹ thuật, không phải sách cuộc thi. Metric private không thể tái tạo cục bộ; sample chính thức vẫn phải được cung cấp cho mỗi lần export production.

## Bắt đầu ở đâu?

| Nhu cầu | Điểm bắt đầu |
|---|---|
| Chạy toàn bộ trên Kaggle/local Jupyter | `notebooks/CASML_R1_end_to_end.ipynb` |
| Chạy B0 thật trên PDF mẫu bằng một lệnh | `python scripts/run_b0_example.py` (sau khi cài `requirements.txt`) |
| Chỉ sửa generation trên cache đã có | Chạy lại section 4–5 trong cùng notebook |
| Chỉ đổi quy ước references/CSV | Chạy lại section 5 trong cùng notebook |
| Kiểm tra ngay, không tải model | Demo bên dưới |
| Đọc hợp đồng dữ liệu | `docs/CONTRACTS.md` |
| Biết thay phần nào thì chạy lại phần nào | `docs/EXPERIMENTS.md` |
| Kiểm tra nguồn tham khảo và giới hạn | `docs/SOURCES.md` |

## Chạy B0 thật bằng CLI

Python 3.10 trở lên. Chạy các lệnh từ thư mục có `pyproject.toml`. Dùng môi trường Python riêng nếu có thể. Lần đầu cần Internet để cài thư viện và tải hai model; không dùng API trả phí. GPU giúp generation nhanh hơn; chế độ `auto` cũng chạy được trên CPU và cần đủ RAM cho model ở float32.

```bash
python -m pip install -r requirements.txt
```

Chuẩn bị `data/book.pdf`, `data/queries.json` và **sample submission chính thức**. Export production sẽ chặn nếu thiếu sample. `queries.json` nhận danh sách hoặc đối tượng có khóa `queries`:

```json
[
  {"query_id": "Q001", "question": "What is working memory?"}
]
```

ID có thể là chuỗi hoặc số nguyên, được chuẩn hóa thành chuỗi và phải duy nhất. Hãy giữ đúng ID gốc của cuộc thi; không tự đánh số lại.

```bash
python -m casml_b0 prepare --pdf data/book.pdf --config configs/prepare.yaml --out artifacts/corpus_printed_v2
python -m casml_b0 index --corpus artifacts/corpus_printed_v2 --config configs/index.yaml --out artifacts/index_bge_v2
python -m casml_b0 retrieve --index artifacts/index_bge_v2 --queries data/queries.json --config configs/retrieve.yaml --out artifacts/retrieval_hybrid_r1
python -m casml_b0 generate --retrieval artifacts/retrieval_hybrid_r1 --config configs/generate.yaml --out runs/hybrid_r1_qwen15b
python -m casml_b0 export --run runs/hybrid_r1_qwen15b --queries data/queries.json --config configs/export.yaml --sample data/sample_submission.csv --pdf data/book.pdf --out outputs/hybrid_r1_qwen15b
```

`--sample` là bắt buộc với `configs/export.yaml`: bộ xuất đối chiếu chính xác thứ tự cột và toàn bộ ID trước khi tạo submission. `--pdf` dùng để xác minh SHA256 và đính kèm sách cạnh báo cáo audit. Export và generation không cần mở index.

Kết quả:

- `outputs/hybrid_r1_qwen15b/submission.csv`: `ID,context,answer,references`.
- `outputs/hybrid_r1_qwen15b/audit.html`: câu hỏi, đáp án, **context thực tế đã đưa vào LLM**, tên mục, trang PDF, trang in, ID chunk và vị trí ký tự.
- `outputs/hybrid_r1_qwen15b/book.pdf`: bản sách dùng để bấm mở nguồn ở đúng trang, nếu truyền `--pdf`.
- `outputs/hybrid_r1_qwen15b/validation.json`: kết quả kiểm tra cấu trúc và nguồn; không phải điểm CASML.
- `runs/hybrid_r1_qwen15b/predictions.jsonl`: đáp án, evidence, prompt thực tế, token và trạng thái từng câu.

Giải nén rồi mở `audit.html` bằng trình duyệt để kiểm tra nguồn; giữ `book.pdf` cùng thư mục. Khả năng nhảy đến `#page=N` phụ thuộc trình đọc PDF của trình duyệt.

## Ranh giới các phần

| Bước | Mã chính | Đọc | Ghi |
|---|---|---|---|
| Đọc sách, làm sạch, chia chunk | `prepare.py` | PDF + cấu hình + page map tùy chọn | Corpus: pages/chunks/page map |
| Embedding, xây index | `embedding.py`, `indexing.py` | Corpus | Index + bản sao chunks |
| Retrieval | `retrieval.py` | Index + queries | Cache chứa toàn văn top 30 ứng viên hybrid đã rerank |
| Đóng gói context, tạo sinh | `context.py`, `llm.py`, `generation.py` | **Chỉ cache retrieval** + cấu hình/prompt | Run generation và checkpoint |
| References, CSV, audit | `exporting.py` | Run generation + queries + sample/PDF tùy chọn | CSV và báo cáo nguồn |

Không truyền đối tượng retriever hoặc embedding model vào LLM. Không gọi retrieval từ generation. `__init__.py` và CLI không tự import model nặng; dependency của mỗi bước nằm trong `requirements-*.txt` riêng.

Cache đã lưu cả text nên có thể chép **nguyên thư mục `retrieval_hybrid_r1/`** sang máy hoặc Kaggle session khác, gồm `manifest.json` và `retrieval.jsonl`. Khi chỉ chạy generation:

```bash
python -m pip install -r requirements-generation.txt
python -m casml_b0 generate --retrieval /path/to/retrieval_hybrid_r1 --config configs/generate.yaml --out runs/hybrid_r1_qwen15b
```

Đổi cấu hình/prompt thì dùng tên `--out` mới. Cùng cấu hình, cùng mã và cùng input có thể chạy lại để tái sử dụng kết quả hoặc tiếp tục checkpoint. Chương trình từ chối trộn kết quả của hai cấu hình trong một thư mục.

`context_top_k` mặc định 4, độc lập với `retrieve.top_k` mặc định 30. Có thể đổi context packing mà không retrieval lại, miễn không vượt số ứng viên trong cache. Giới hạn token và loại trùng vẫn có thể làm số đoạn thực tế ít hơn.

## Cấu hình mặc định

| Phần | B0 |
|---|---|
| Chunk | Tối đa 300 token theo tokenizer BGE, overlap 50; không cắt qua trang PDF |
| Nguồn mục | Bookmark/TOC gần đúng theo trang; page map xác minh tay được ưu tiên |
| Embedding | BGE-small-en-v1.5, vector chuẩn hóa, tiền tố chỉ cho query |
| Retrieval | FAISS/BGE-small top 100 + BM25 top 100, weighted RRF, cross-encoder rerank top 50 → lưu top 30 |
| Context | Tối đa 4 chunk; 1.900 token theo tokenizer của LLM; bỏ trùng/overlap lớn |
| LLM | Qwen2.5-1.5B-Instruct, greedy, tối đa 384 token mới, EOS guard |
| Token safety | Giới hạn tổng 4.096 token; đếm chat template thật; không âm thầm cắt câu hỏi |
| References | Lấy từ metadata của những chunk thực sự vào prompt; LLM không tự viết số trang |

Revision của cả hai model được ghim trong YAML. Khi đổi LLM, đổi cả `model_name` và `revision` trong `generate.yaml` cho đúng model mới. Nếu đổi embedding, sửa cấu hình riêng trong `index.yaml`; nếu đổi tokenizer chia chunk, cập nhật `prepare.yaml` tương ứng. Không giữ revision của model cũ.

## Trang PDF và trang in

`pdf_page` luôn tính từ 1. Với cấu hình CASML production, `printed_page = pdf_page - 12` cho các trang nội dung; PDF page 20 tương ứng textbook page 8. PDF label hoặc `page_map_override.csv` được ưu tiên khi có.

Sau prepare, đọc `page_map.csv` và `report.json`. Với PDF không có bookmark, mục sẽ để trống thay vì bịa tên mục. Với sách có bookmark, ranh giới mục trong cùng một trang vẫn có thể gần đúng. Có thể tạo file override:

```csv
pdf_page,printed_page,section_path
13,1,1 Introduction/1.1 Overview
14,2,1 Introduction/1.1 Overview
```

Chạy prepare với `--page-map-override data/page_map_override.csv` và một `--out` mới, rồi dựng lại các artifact phía sau. Mỗi dòng override áp dụng cho **một trang**, không tự kéo dài đến trang tiếp theo. Để trống `printed_page` nghĩa là chưa biết. Mã hiện chưa gán nhiều section khác nhau cho các đoạn trong cùng một trang; nếu cần độ chính xác đó, cải thiện riêng `prepare.py`.

`configs/prepare.yaml` ánh xạ trang sách bằng offset `-12` của PDF cuộc thi (ví dụ PDF page 20 → textbook page 8), và `configs/export.yaml` xuất `page_mode: printed`. Nếu PDF thay đổi hoặc kiểm tra audit cho thấy offset sai, phải cung cấp `page_map_override.csv` đã kiểm chứng và tạo chuỗi corpus/index/retrieval/run mới.

## Kaggle

1. Import/upload `notebooks/CASML_R1_end_to_end.ipynb` vào Kaggle; thêm dữ liệu CASML chính thức vào Input.
2. Bật Internet và GPU nếu có. Cell setup tự clone `https://github.com/lighth-gh/casml-rag.git` vào `/kaggle/working/casml-rag`; Internet cũng cần cho lần tải model đầu.
3. Khi chạy lại notebook, mã đã clone được dùng lại. Đặt `UPDATE_REPO = True` nếu muốn `git pull --ff-only`, hoặc đặt `ROOT_OVERRIDE` nếu muốn dùng một bản project khác.
4. Cell cấu hình tự dò PDF sách, `queries.json` và `sample_submission.csv` trong `/kaggle/input`. Notebook dừng ngay nếu thiếu sample chính thức; nếu có nhiều ứng viên, đặt `PDF_OVERRIDE`, `QUERIES_OVERRIDE` hoặc `SAMPLE_OVERRIDE` tới file chính xác. Notebook không tự chọn PDF có tên dạng overview/instructions/rules/guide.
5. Chạy lần lượt, đọc báo cáo prepare/audit rồi đối chiếu sample và quy ước references chính thức.
6. Lưu thư mục cache retrieval làm output/dataset riêng. Với thử nghiệm LLM mới, trỏ `RETRIEVAL` đến cache đó rồi chạy lại section 0, 4 và 5 trong cùng notebook.

Notebook chạy mỗi bước bằng process riêng để giải phóng model sau khi bước kết thúc. Nó không ghi vào `/kaggle/input`. Nếu làm hoàn toàn offline, chuẩn bị model snapshot trước và sửa YAML dùng đường dẫn local với `local_files_only: true`.

## Chạy demo nhanh

Chế độ này kiểm tra luồng và metadata bằng word hashing + trích câu đơn giản; **không dùng model RAG thật**. Nó không được phép xuất bằng cấu hình production mặc định.

```bash
python -m pip install -e .
python scripts/run_demo.py
python -m unittest discover -s tests -v
```

PDF/queries mẫu đã có trong `examples/`. `scripts/make_demo.py` có thể tạo lại. Demo tạo `outputs/demo/`. Để chạy **B0 thật trên dữ liệu mẫu**, cài `requirements.txt` rồi chạy `python scripts/run_b0_example.py`. Kết quả nằm ở `outputs/example_b0/`. Bản kiểm tra đã chạy kèm theo gói nằm ở `examples/verified/`; đây là kết quả trên sách mẫu, không phải submission cho contest.

## Chạy lại, lỗi và giới hạn

- Mỗi run generation mới tạo `runs/<RUN_NAME>/diagnosis.json` ngay khi khởi tạo run hợp lệ, rồi cập nhật sau mỗi lần sinh và mỗi câu. File chứa cấu hình/prompt, phiên bản môi trường, câu hỏi, đáp án đầy đủ, context/evidence, số token, lịch sử retry và traceback lỗi; có thể gửi riêng file này để chẩn đoán. Các trạng thái gồm `running`, `completed`, `failed`, `interrupted`; summary liệt kê câu lỗi, bị cắt, thiếu evidence và chưa chạy. File vẫn được lưu khi lỗi tải model, lỗi một câu hoặc Ctrl+C; nếu tiến trình bị kill cứng thì chỉ còn snapshot gần nhất, có thể vẫn ghi `running`. Lỗi input/config trước khi khởi tạo run không tạo artifact. Notebook hiển thị link diagnosis cả khi generation báo lỗi. Run hoàn tất đưa checksum file vào manifest. `generation_ready_for_export` chỉ kiểm tra điều kiện generation, không xác nhận chất lượng đáp án hay schema contest. Dùng run mới mặc định `hybrid_r2_qwen15b_t384_eos_guard`; không trộn với checkpoint của run lỗi cũ.
- Generation lưu mỗi câu ngay sau khi xử lý. Nếu lỗi runtime, chạy lại đúng lệnh để giữ câu thành công và thử lại câu lỗi. Nếu đổi code/config/prompt, tạo run mới.
- `generate --limit 3` chỉ kiểm tra nhanh 3 câu đầu; dùng một thư mục run riêng. Export sẽ từ chối dùng run thiếu câu với bộ queries đầy đủ.
- Qwen dùng rõ hai stop token chính thức `<|im_end|>`/`<|endoftext|>` và SDPA. Đáp án tối đa 384 token; nếu chạm trần, generation chỉ thử lại một lần ở cùng ngân sách với prompt ngắn và kiểm soát lặp, không tăng lên 1024/2048 token.
- Nếu lần thử lại vẫn bị cắt, production dừng ngay ở câu đó (`abort_after_consecutive_length_limited: 1`) thay vì đốt GPU cho 50 câu. Checkpoint bị cắt không được coi là hoàn tất và sẽ được sinh lại sau khi sửa. Export vẫn chặn mọi đáp án `length_limited`.
- Không tái sử dụng `retrieval_v1` của run 0.278: hybrid retrieval đã chuyển sang `retrieval_hybrid_r2`, và page mapping `-12` cần chuỗi corpus/index mới để metadata nguồn nhất quán. Cache `retrieval_hybrid_r1` hợp lệ vẫn có thể trỏ trực tiếp vào section generation, nhưng không ghi đè nó bằng mã/config mới.
- Artifact có chữ ký cấu hình, mã liên quan và input; file output có SHA256. Cache bị sửa tay sẽ bị từ chối. Đây là kiểm tra tính toàn vẹn cục bộ, không phải chữ ký xác thực bên thứ ba.
- Prepare/index/retrieve/export chưa resume từng phần; nếu bị ngắt giữa bước, dùng thư mục output mới hoặc xóa đúng thư mục chưa hoàn tất sau khi kiểm tra. Generation có resume theo câu.
- PDF scan cần OCR trước; baseline này chưa có OCR. Sách nhiều cột/bảng có thể cần cải thiện text extraction. Kiểm tra `empty_pages`, vài trang mẫu và các câu cần bảng trước khi benchmark.
- Có nguồn để mở lại không đảm bảo từng ý của đáp án đều được nguồn hỗ trợ. B0 chưa có verifier, bộ nhãn đánh giá retrieval hoặc metric CASML cục bộ. Xem hướng kiểm tra độc lập trong `docs/EXPERIMENTS.md`.
