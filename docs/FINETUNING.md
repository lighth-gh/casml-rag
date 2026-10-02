# Fine-tune generator Qwen2.5-1.5B-Instruct

Bước tùy chọn dùng LoRA, giữ nguyên retrieval. Model/revision và prompt lấy từ
`configs/generate.yaml`; thông số train nằm trong `configs/finetune.yaml`.
Cần CUDA GPU và môi trường cài được `requirements-finetune.txt` (nên dùng Python
3.10–3.12 như Kaggle). Notebook mặc định tắt fine-tune khi chưa có dữ liệu.

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
riêng trước khi huấn luyện thực tế. Bước này không tự sinh nhãn khi chưa có dữ liệu.

Nếu không truyền validation riêng, chương trình chia holdout 10% theo seed 42,
với ít nhất một dòng mỗi tập. ID và câu hỏi trùng bị từ chối; nếu truyền hai file,
chương trình cũng kiểm tra ID/câu hỏi trùng giữa hai tập (không phát hiện paraphrase).

## Chạy trong notebook

Trong `notebooks/CASML_R1_end_to_end.ipynb`, section **3b**:

1. Đặt `TRAIN_JSONL` tới dữ liệu đã chuẩn bị; có thể đặt `VALIDATION_JSONL` riêng.
2. Đặt `FINETUNE_ENABLED = True`, chọn `FINETUNE_OUT` mới cho mỗi cấu hình/dataset mới.
3. Chạy section 3b rồi section 4–5. Section 4 tự chọn model vừa train, dùng run/output riêng.

Fine-tune không cần index/PDF. Có thể chỉ chạy setup section 0 và section 3b khi
JSONL đã chứa context. Để dùng artifact từ session khác, chép **toàn bộ** thư mục
fine-tune rồi đặt `FINETUNED_ARTIFACT` trong section 4; cell này cập nhật đường dẫn
model/prompt khi chuyển artifact. Muốn trở về baseline, đặt `FINETUNED_ARTIFACT = None`.

## Chạy bằng CLI

```bash
python -m pip install -r requirements-finetune.txt
python -m casml_b0 finetune --train data/train.jsonl --validation data/validation.jsonl --config configs/finetune.yaml --out artifacts/qwen15b_lora_v1
python -m casml_b0 generate --retrieval artifacts/retrieval_hybrid_r1 --config artifacts/qwen15b_lora_v1/generate.yaml --out runs/qwen15b_finetuned_v1
```

Bỏ `--validation` để chia holdout tự động. Đường dẫn train/out tính từ thư mục
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
và dùng `--out` mới; chưa hỗ trợ resume một lần train bị gián đoạn. Chạy lại artifact
đã hoàn tất với cùng config/input/code sẽ tái sử dụng kết quả.

Tham khảo API: [Transformers Trainer 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer)
và [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
