# Những gì đã kiểm tra trên bản B0

## Tích hợp model thật

Kết quả model thật dưới đây là bản kiểm tra lịch sử của B0. R1 hiện tại được kiểm tra bằng contract tests/mocks; lần chạy thật Qwen 1.5B + cross-encoder cần thực hiện trên Kaggle GPU với dữ liệu cuộc thi.

B0 trước đây đã chạy đủ 5 bước trên CPU với PDF mẫu 4 trang và 3 câu hỏi đi kèm:

- BGE-small-en-v1.5 → FAISS → Qwen2.5-0.5B-Instruct.
- Cả 3 câu hoàn tất, không lỗi runtime, không chạm giới hạn token; CSV có đủ ID và references đọc lại được.
- PDF đính kèm audit khớp SHA256 của sách đã index.
- Thời gian quan sát trong môi trường kiểm tra: prepare 6,80 giây; index 7,44 giây; retrieve 6,93 giây; generate 20,34 giây; export 0,30 giây. Không tính tải model/cài thư viện; không dùng để dự báo thời gian toàn bộ contest.
- Phiên bản thư viện và artifact ID nằm trong `examples/verified/verification.json` và các manifest/report đi kèm.

**Chất lượng chưa phải tiêu chí đã đạt:** Q001/Q002 cho đáp án có nội dung phù hợp với đoạn mẫu. Q003 chỉ trả lời “independent variable”, thiếu định nghĩa mặc dù nguồn Research có định nghĩa. Giữ nguyên kết quả này làm mốc B0; sau này sửa prompt/model tại generation trên cùng cache để đo cải thiện. Không tự sửa tay đáp án trước khi đóng gói.

## 12 kiểm tra tự động

1. PDF → chunks → cache → generation → CSV; span text khớp trang.
2. Xóa corpus/index và chặn import FAISS/SentenceTransformer/retrieval: generation vẫn chạy bằng backend demo.
3. Đổi cấu hình generation phải tạo run mới; byte cache retrieval giữ nguyên.
4. Đổi nội dung file prompt không trộn checkpoint cũ.
5. Giả lập ngắt giữa chừng: giữ câu thành công, chạy lại câu lỗi và câu chưa chạy.
6. Cache bị sửa bị phát hiện qua checksum trước generation.
7. Context vừa ngân sách; giữ nguyên câu hỏi và text từng chunk.
8. References chỉ lấy evidence thực sự dùng; kiểm tra số trang in và lỗi thiếu map.
9. Run thiếu câu không được xuất cho toàn bộ queries.
10. Giữ thứ tự ID của sample; từ chối PDF audit khác sách gốc.
11. Backend demo không được xuất nhầm bằng cấu hình B0 thật.
12. ID trùng sau chuẩn hóa bị từ chối.

12/12 đã qua. Các kiểm tra này dùng backend nhẹ để cô lập lỗi hợp đồng, không dùng điểm chất lượng của model để xác nhận đúng/sai.

## Kiểm tra đóng gói

- Cài project bằng `pip install --no-deps -e .` thành công.
- Sáu notebook hợp lệ theo nbformat và các code cell biên dịch được. Luồng CLI đã chạy thực tế; chưa chạy notebook trong phiên Kaggle của người dùng.
- PDF mẫu đã render và kiểm tra trực quan; có nhúng font để tránh lỗi hiển thị.
- Gói không chứa model weights, API key hoặc dữ liệu sách của cuộc thi.

Chưa chạy R1 hiện tại trên full corpus CASML trong môi trường local này và không thể tái tạo metric private. File CSV trong `examples/` chỉ phục vụ kiểm tra phần mềm, không để nộp contest.
