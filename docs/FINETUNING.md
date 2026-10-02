# Fine-tune generator Qwen2.5-1.5B-Instruct

Bước tùy chọn dùng LoRA, giữ nguyên retrieval. Model/revision và prompt lấy từ
`configs/generate.yaml`; thông số train nằm trong `configs/finetune.yaml`.
Cần CUDA GPU và môi trường cài được `requirements-finetune.txt` (nên dùng Python
3.10–3.12 như Kaggle). Notebook mặc định bật fine-tune; không có JSONL thì tự
tạo dữ liệu tổng hợp từ corpus sách ở section 1.

## Cuộc thi có JSONL train không?

Theo phần **Dataset → What’s Provided** của Overview mà người dùng cung cấp,
cuộc thi có `Queries.json`, câu hỏi test không kèm đáp án, sách và sample submission.
Không thấy công bố bộ Q&A train có nhãn ở phần này. `Queries.json` không thể
đổi đuôi thành JSONL rồi dùng để supervised fine-tune vì thiếu context/answer.
JSONL là định dạng pipeline này nhận, không phải file cuộc thi hứa cung cấp.

Tham khảo: [CASML Overview](https://www.kaggle.com/competitions/casml-generative-ai-hackathon/overview).
Trang Kaggle được render động nên công cụ đọc web không lấy được danh sách file
đầy đủ; kết luận trên dựa vào nội dung Overview người dùng gửi.

## Không có nhãn: tạo dữ liệu từ sách

Lệnh `build-sft` đọc `chunks.jsonl` từ artifact corpus đã được kiểm checksum,
sử dụng Qwen gốc sinh một câu hỏi cùng đáp án trích nguyên văn từ mỗi đoạn.
Nó không nhận queries test, predictions hoặc sample submission. Dùng cùng model
1.5B sẵn có, không gọi API tạo sinh bên ngoài.

- Mặc định xét tối đa 600 chunk, giữ tối đa 200 cặp; cần ít nhất 20 cặp và cả hai tập không rỗng.
- Chỉ giữ đáp án 12–100 từ xuất hiện nguyên văn, liên tiếp trong chunk nguồn.
- Loại JSON hỏng, đáp án bị cắt, câu hỏi/đáp án trùng, câu hỏi thiếu độc lập.
- Chia trang nguồn train/validation theo seed trước khi chọn chunk; cùng trang không xuất hiện ở hai tập.
- Lưu nguồn, trang, loại nhãn tổng hợp, lý do loại trong report và checkpoint từng lần sinh.

Trích đúng nguồn **không chứng minh đáp án trả lời đúng câu hỏi**. Cần kiểm tra
mẫu trong `train.jsonl`/`validation.jsonl` và so sánh model với baseline. Chia theo
trang vẫn có thể để các chủ đề gần nhau xuất hiện ở hai tập; loss validation
tổng hợp không đo chất lượng trên ground truth cuộc thi. Không cam kết tăng điểm.

Nguồn sách: OpenStax, *Psychology 2e*, CC BY 4.0 theo phần Dataset License
trong Overview được cung cấp; [trang sách](https://openstax.org/details/books/psychology-2e).
Dữ liệu tạo ra gồm câu hỏi tổng hợp và trích đoạn có metadata từ sách; giữ nguồn
và thông tin giấy phép khi chia sẻ.

## Chuẩn bị JSONL

Mỗi dòng là một đối tượng độc lập gồm bốn trường bắt buộc:

```json
{"query_id":"train_001","question":"What is retrieval?","context":"[E1] PDF page 2 | Memory\nRetrieval is the process of accessing information stored in memory.","answer":"Retrieval is the process of accessing information stored in memory."}
```

`context` là evidence hỗ trợ đáp án, nên dùng đúng định dạng `[E1] PDF page ... | ...`
như pipeline generation. Có thể lấy context đã đóng gói từ cache/run để gán nhãn,
nhưng `answer` phải là đáp án chuẩn được kiểm tra với evidence. Không dùng câu hỏi
test/đáp án tự dự đoán làm dữ liệu train. Chia train/validation theo chủ đề hoặc
nguồn trước khi đưa vào pipeline nếu có nhiều câu hỏi gần giống nhau.

`examples/finetune_train.jsonl` chứa ba câu tổng hợp để minh họa schema, không phải
dữ liệu cuộc thi và không đủ để cải thiện chất lượng model. Thay bằng bộ dữ liệu
riêng trước khi huấn luyện có nhãn thực tế. Luồng tự tạo dữ liệu từ sách không dùng file mẫu này.

Nếu không truyền validation riêng, chương trình chia holdout 10% theo seed 42,
với ít nhất một dòng mỗi tập. ID và câu hỏi trùng bị từ chối; nếu truyền hai file,
chương trình cũng kiểm tra ID/câu hỏi trùng giữa hai tập (không phát hiện paraphrase).

## Chạy trong notebook

Trong `notebooks/CASML_R1_end_to_end.ipynb`, section **3b**:

1. Chạy section 0 và 1 để có corpus sách; bật GPU trên Kaggle.
2. Giữ `FINETUNE_ENABLED = True` và `TRAIN_JSONL = None`: tự tạo dữ liệu và train.
3. Chạy section 4–5 để dùng model đã train. Nếu chưa có retrieval, chạy section 2–3 trước section 4.

Đặt `TRAIN_JSONL`/`VALIDATION_JSONL` nếu đã có dữ liệu riêng để bỏ qua bước tạo
nhãn tổng hợp. Notebook dùng 1 epoch và learning rate 5e-5 cho lần thử đầu;
chỉnh ngay trong cell fine-tune nếu cần. `FINETUNE_ENABLED = False` chạy baseline.
Mỗi stage chạy process riêng, giải phóng model trước stage kế tiếp.

Khi đổi cấu hình/dữ liệu, dùng `SFT_DATA` và `FINETUNE_OUT` mới. Nếu quá ít cặp
hợp lệ, bước tạo dữ liệu dừng và ghi lý do; không tự tiếp tục train dữ liệu rỗng.

Fine-tune không cần index/PDF khi đã có JSONL; luồng tự tạo JSONL cần corpus từ section 1.
Để dùng artifact từ session khác, chép **toàn bộ** thư mục
fine-tune rồi đặt `FINETUNED_ARTIFACT` trong section 4; cell này cập nhật đường dẫn
model/prompt khi chuyển artifact. Muốn trở về baseline, đặt `FINETUNED_ARTIFACT = None`.

## Chạy bằng CLI

```bash
python -m pip install -r requirements-finetune.txt
python -m casml_b0 build-sft --corpus artifacts/corpus_printed_v2 --config configs/synthetic.yaml --out artifacts/book_synthetic_sft_v1
python -m casml_b0 finetune --train artifacts/book_synthetic_sft_v1/train.jsonl --validation artifacts/book_synthetic_sft_v1/validation.jsonl --config configs/finetune.yaml --out artifacts/qwen15b_lora_v1
python -m casml_b0 generate --retrieval artifacts/retrieval_hybrid_r1 --config artifacts/qwen15b_lora_v1/generate.yaml --out runs/qwen15b_finetuned_v1
```

Với dữ liệu tổng hợp, luôn truyền `validation.jsonl` đã tạo để giữ phân chia theo
trang. Có thể chỉnh `configs/finetune.yaml` thành 1 epoch, learning rate 0.00005
như notebook. Với dữ liệu tự gán nhãn, thay đường dẫn train/validation bằng file riêng.

Bỏ `--validation` với dữ liệu riêng để chia holdout tự động. Đường dẫn train/out tính từ thư mục
chạy lệnh; `generation_config` tính từ thư mục chứa YAML fine-tune. File
`generate.yaml` xuất ra dùng đường dẫn tuyệt đối cho model; khi chuyển artifact,
dùng cell section 4 để tạo config runtime mới, không sửa file trong artifact.

Đầu ra gồm `adapter/`, model đã merge trong `model/`, prompt, `generate.yaml`,
`report.json`, checkpoints và manifest/checksum. Generator dùng model đã merge nên
không cần PEFT ở phiên inference. Không chỉnh model/file trong artifact sau train;
generation kiểm tra checksum và đưa ID fine-tune vào định danh cache.

Loss chỉ tính trên câu trả lời assistant và token kết thúc lượt; prompt/context
và padding có nhãn `-100`. Ví dụ vượt `max_length` bị từ chối, không âm thầm cắt
câu hỏi/đáp án. Mặc định 2.048 token, batch size 1, tích lũy gradient 8 bước, LoRA
rank 16, learning rate 2e-4, 3 epoch, gradient checkpointing. `dtype: auto` dùng
BF16 nếu GPU hỗ trợ, FP16 nếu không. Đây là LoRA không lượng tử hóa 4-bit.

`report.json` ghi loss trước train và loss validation của checkpoint tốt nhất,
cùng ID từng tập. Loss này không phải điểm CASML; cần so sánh baseline và model
fine-tune trên dữ liệu giữ riêng, cùng retrieval/prompt. Khi lỗi/OOM, sửa cấu hình
và dùng `--out` mới; chưa hỗ trợ resume một lần train LoRA bị gián đoạn.
Riêng `build-sft` có checkpoint: chạy lại cùng input/config/out để tiếp tục sinh
dữ liệu còn thiếu mà không sinh lại những chunk đã có kết quả. Chạy lại artifact
đã hoàn tất với cùng config/input/code sẽ tái sử dụng kết quả.

Tham khảo API: [Transformers Trainer 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer)
và [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
