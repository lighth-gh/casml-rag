# Hợp đồng giữa các bước

Schema hiện tại: `casml-b0/v1`. Chỉ thay module thuộc bước cần cải thiện; giữ hợp đồng đầu ra để bước sau tiếp tục dùng được. Khi đổi schema, nâng phiên bản và viết adapter rõ ràng.

## Artifact và dấu vết

Mỗi thư mục output có `manifest.json` với stage, artifact ID, cấu hình hiệu lực, hash input, hash mã liên quan, trạng thái `complete` và checksum từng file. Bước sau chỉ nhận artifact hoàn tất có checksum đúng.

Chữ ký không hash toàn bộ project: đổi `llm.py`, `context.py`, `generation.py` hoặc nội dung prompt không làm thay đổi ID corpus/index/retrieval cũ. Hai module chung `artifacts.py`, `contracts.py` nằm trong chữ ký mọi bước vì thay chúng có thể đổi hợp đồng dữ liệu.

Chữ ký không bao gồm toàn bộ trạng thái driver/hardware; `report.json` ghi phiên bản thư viện. Greedy và seed giúp giảm biến thiên nhưng không đảm bảo bitwise giống nhau giữa mọi GPU/PyTorch.

## 1. Corpus

`pages.jsonl`: mỗi trang có `doc_id`, `source_pdf`, `pdf_page`, `printed_page`, `section_path`, `section_method`, `text`, `text_sha256`.

`chunks.jsonl`: cùng metadata, thêm `chunk_id`, `char_start`, `char_end`, `chunk_token_count`. Span ký tự trỏ vào **text đã chuẩn hóa của trang trong pages.jsonl**, không trỏ vào byte PDF hoặc bounding box. Luôn có:

`chunk.text == page.text[char_start:char_end]`

`doc_id` là SHA256 byte PDF. `text_sha256` là hash qua bộ serialize chuẩn hóa của dự án (`artifacts.digest`), không phải hash trực tiếp UTF-8 của chuỗi. `chunk_id` ổn định theo PDF, trang, span và text.

`section_method` phân biệt `user_override`, `toc_page_approximation`, `unknown`. Nguồn mục gần đúng phải hiện rõ trong audit; không được gắn nhãn là đã kiểm chứng thủ công.

## 2. Index

`chunks.jsonl` được sao chép vào artifact index. Vector thứ i tương ứng dòng thứ i; FAISS lưu IndexFlatIP trên vector chuẩn hóa. Manifest giữ cấu hình embedding để retrieval sử dụng đúng model và đúng query prefix.

Thay embedding model có thể tái dùng corpus khi chunk vẫn vừa giới hạn tokenizer mới. Nếu vượt, embedding sẽ báo lỗi thay vì tự cắt text; lúc đó cần đổi chunking và chạy lại từ prepare.

## 3. Retrieval cache: ranh giới chính

Mỗi dòng `retrieval.jsonl` có:

| Trường | Ý nghĩa |
|---|---|
| `schema` | Phiên bản hợp đồng |
| `query_id`, `question` | Câu hỏi đầy đủ, không phụ thuộc file queries ở generation |
| `candidates` | Danh sách ứng viên đã sắp xếp theo retrieval |
| `candidates[].rank`, `score` | Thứ hạng và điểm tương đồng |
| `candidates[].chunk_id`, `text`, `text_sha256` | Nội dung tự chứa, kiểm tra được |
| `candidates[].doc_id`, `source_pdf` | Sách gốc |
| `candidates[].pdf_page`, `printed_page` | Hai hệ số trang riêng |
| `candidates[].section_path`, `section_method` | Mục và cách xác định mục |
| `candidates[].char_start`, `char_end` | Span trong trang đã chuẩn hóa |

Để chuyển cache sang máy khác chỉ cần **manifest + retrieval.jsonl**, không cần pickle Python, index, embedding hay PDF. Giữ manifest để bước generation kiểm tra checksum và tham chiếu cấu hình retrieval.

R1 dùng dense + BM25 và weighted reciprocal-rank fusion, sau đó cross-encoder rerank. Generation đọc thứ tự `candidates`, không giả định công thức score. Mỗi candidate giữ `dense_*`, `bm25_*`, `fusion_score` và `reranker_score` để audit; `retrieval_score` là điểm xếp hạng cuối.

## 4. Generation

`context.py` chọn chunk trong cache, loại trùng và đóng gói dưới ngân sách tokenizer của LLM. Không gọi truy hồi mới. B0 lấy nguyên chunk để giữ span nguồn; chunk không vừa ngân sách được bỏ qua, không cắt câu hỏi.

`llm.py` có giao diện generator gồm `count_text`, `count_messages`, `context_window`, `generate`. Có thể thay model backend tại đây mà không sửa `retrieval.py`.

`predictions.jsonl` lưu cả question, answer, evidence đã dùng, context đúng như đưa vào user message, messages đầy đủ, token, thời gian, trạng thái, finish reason, run ID và checksum bản ghi. Mỗi câu có checkpoint riêng; câu `error` được chạy lại, câu thành công được tái sử dụng nếu chữ ký không đổi.

Không chỉnh text/metadata của evidence tại generation. Muốn đổi cách trình bày cho LLM, sửa `render_context` và packing; muốn sửa text nguồn, sửa corpus. B0 chỉ chấp nhận bằng chứng từ sách, không thêm web hay dữ liệu ngoài.

## 5. Export và references

Schema CSV hỗ trợ: `ID,context,answer,references`. `references` là JSON string có `sections` và `pages`. CSV dùng thư viện chuẩn để quote đúng xuống dòng/dấu phẩy/dấu nháy; exporter đọc lại CSV sau khi ghi.

Trước khi một answer được chấp nhận, grounding validator yêu cầu mọi số, năm và tên riêng nhận diện được phải xuất hiện trong packed evidence. Generation đưa bản nháp cũ vào một edit retry tối đa 224 token. Nếu vẫn còn vi phạm, fallback bảo thủ thử các bản nháp và bỏ nguyên câu hoặc ngoặc chứa token không có nguồn; mọi thay đổi được ghi trong `grounding_repair`. Chỉ vi phạm không sửa được mới có trạng thái `unsupported_claims` và chặn export. Đây là kiểm tra token support, không chứng minh quan hệ giữa các dữ kiện là đúng.

References chỉ dùng **packed evidence**, không dùng toàn bộ 20 candidates và không lấy số trang do LLM sinh. CSV context khớp context generation. Với sample chính thức, giữ thứ tự cột/ID của sample và yêu cầu tập ID trùng khớp hoàn toàn.

Page mode chỉ thuộc export. Cấu hình production ánh xạ `printed_page` ngay từ prepare bằng offset `-12`; sửa offset hoặc page map cần một chuỗi artifact mới để metadata evidence nhất quán.

Audit tự chứa HTML, không gọi dịch vụ ngoài. Nếu đính kèm PDF, exporter kiểm tra SHA256 bằng doc_id rồi sao chép cạnh HTML; mỗi evidence có liên kết `book.pdf#page=N`.
