# Nguồn tham khảo và các điều chưa xác minh

## CASML

- [Trang cuộc thi](https://www.kaggle.com/competitions/casml-generative-ai-hackathon/overview).
- [Thông báo giải chính thức CASML 2024](https://casml.cc/best-papers-and-posters-awards-2024/): căn cứ xác định hạng nhất ciwrl1 và hạng nhì Vansh Dhar.
- [Notebook công khai của ciwrl1](https://www.kaggle.com/code/ciwrl1/casml-genai-hackaton-advanced-rag): tham khảo cách tổ chức chunk có trang/mục, dense + BM25, reranking, generation và xuất submission.
- [Notebook công khai của Vansh Dhar](https://www.kaggle.com/code/vansh63/casml-rag-2024): tham khảo BGE, reranker, HyDE, instruction model và context/references.

B0 này giữ những phần cần cho baseline kiểm tra được: metadata xuyên suốt, retrieval riêng, context có ngân sách, model instruction và CSV. Các lựa chọn BGE-small, Qwen2.5-0.5B-Instruct, chia chunk theo trang, cache tự chứa, chữ ký artifact và audit HTML là thiết kế của bản starter; không gán chúng cho đội đạt giải. Không tuyên bố tái lập điểm số của hai notebook. BM25 và Qwen 1.5B là hướng nâng cấp sau, chưa tích hợp trong B0.

Các notebook công khai dùng schema `ID,context,answer,references`, với references gồm `sections` và `pages`. Starter triển khai schema đó và kiểm tra thêm nếu bạn cung cấp sample chính thức. Chưa xác minh được metric chính thức và quy ước trang duy nhất cho bộ dữ liệu gốc; mặc định xuất số trang PDF và yêu cầu đối chiếu trước khi nộp.

## Model và thư viện

- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5), revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- [SentenceTransformer API](https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html).
- [PyMuPDF Document API](https://pymupdf.readthedocs.io/en/latest/document.html).
- [FAISS: tìm kiếm cơ bản](https://github.com/facebookresearch/faiss/wiki/Getting-started).
- [Transformers generation](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/text_generation).

Thư viện/model không được đóng gói trong ZIP; được cài/tải khi chạy. Cần tuân theo license riêng của dữ liệu, model và dependency khi tái sử dụng.

## Phạm vi kiểm chứng

Các kiểm tra tự động dùng một PDF ngắn do dự án tự tạo để xác minh hợp đồng, resume, ID, token, references, checksum và khả năng generation tách khỏi retrieval. Chạy model thật trên PDF mẫu chỉ xác minh tích hợp kỹ thuật, không đo hiệu quả trên sách CASML.

Không có truy vấn web trong pipeline. Các URL trên phục vụ tài liệu thiết kế, không được đưa vào corpus hoặc prompt trả lời câu hỏi của cuộc thi.
