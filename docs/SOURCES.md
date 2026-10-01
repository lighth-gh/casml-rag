# Nguồn tham khảo và các điều chưa xác minh

## CASML

- [Trang cuộc thi](https://www.kaggle.com/competitions/casml-generative-ai-hackathon/overview).
- [Thông báo giải chính thức CASML 2024](https://casml.cc/best-papers-and-posters-awards-2024/): căn cứ xác định hạng nhất ciwrl1 và hạng nhì Vansh Dhar.
- [Notebook công khai của ciwrl1](https://www.kaggle.com/code/ciwrl1/casml-genai-hackaton-advanced-rag): tham khảo cách tổ chức chunk có trang/mục, dense + BM25, reranking, generation và xuất submission.
- [Notebook công khai của Vansh Dhar](https://www.kaggle.com/code/vansh63/casml-rag-2024): tham khảo BGE, reranker, HyDE, instruction model và context/references.

R1 giữ metadata xuyên suốt, retrieval riêng, context có ngân sách, model instruction và CSV; đồng thời tích hợp BGE-small + BM25, weighted reciprocal-rank fusion, `cross-encoder/ms-marco-MiniLM-L-12-v2` và Qwen2.5-1.5B-Instruct. Không tuyên bố tái lập điểm số của notebook công khai hay leaderboard private.

Các notebook công khai và yêu cầu cuộc thi dùng schema `ID,context,answer,references`, với references gồm `sections` và `pages`. Một notebook ánh xạ TOC từ PDF sang số trang sách bằng offset `-12`; đối chiếu corpus thực tế cũng cho thấy PDF page 20 là textbook page 8. R1 xuất trực tiếp theo schema đã công bố, kiểm tra đủ ID và parse lại CSV/JSON mà không yêu cầu sample. Metric private không thể tái tạo cục bộ.

## Model và thư viện

- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5), revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- [MS MARCO MiniLM-L12 cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2), revision `7b0235231ca2674cb8ca8f022859a6eba2b1c968`.
- [SentenceTransformer API](https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html).
- [PyMuPDF Document API](https://pymupdf.readthedocs.io/en/latest/document.html).
- [FAISS: tìm kiếm cơ bản](https://github.com/facebookresearch/faiss/wiki/Getting-started).
- [Transformers generation](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/text_generation).

Thư viện/model không được đóng gói trong ZIP; được cài/tải khi chạy. Cần tuân theo license riêng của dữ liệu, model và dependency khi tái sử dụng.

## Phạm vi kiểm chứng

Các kiểm tra tự động dùng một PDF ngắn do dự án tự tạo để xác minh hợp đồng, resume, ID, token, references, checksum và khả năng generation tách khỏi retrieval. Chạy model thật trên PDF mẫu chỉ xác minh tích hợp kỹ thuật, không đo hiệu quả trên sách CASML.

Không có truy vấn web trong pipeline. Các URL trên phục vụ tài liệu thiết kế, không được đưa vào corpus hoặc prompt trả lời câu hỏi của cuộc thi.
