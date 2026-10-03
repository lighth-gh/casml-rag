# Fine-tune Qwen2.5-1.5B-Instruct — workflow v2

Cập nhật 03/10/2026. Notebook phát triển mặc định bật `FINETUNE_ENABLED = True`.
Section 4 mặc định dùng `WORK / "artifacts/qwen15b_selected_v2"` khi fine-tuning bật.
Thiếu dữ liệu đã duyệt hoặc model đã chọn thì notebook dừng với hướng dẫn, không tự chạy baseline.
Train xong chỉ tạo **candidates**; cần hoàn tất đánh giá và merge ở 3d trước inference.
Dữ liệu và điểm dưới đây là bộ đánh giá nội bộ, không phải gold/scorer chính thức của Kaggle.

## Trình tự trong notebook

Dùng `notebooks/CASML_R1_end_to_end.ipynb` để phát triển. Section 3b có ba công tắc độc lập:

1. Đặt `FINETUNE_ENABLED=False`, `BUILD_DRAFTS=True`: tạo câu hỏi, đáp án diễn đạt lại và trích đoạn hỗ trợ từ sách.
2. Section 3c, `ATTACH_DRAFT_CONTEXT=True`: retrieve các câu hỏi vừa tạo rồi đóng gói
   context bằng đúng `pack_context` của inference. **Không dùng queries của cuộc thi.**
3. Copy `context_drafts_v2/drafts.jsonl` ra file annotation riêng, đọc sách và duyệt theo schema dưới đây.
   Quay lại section 3b, đặt `ANNOTATIONS_JSONL` và `PREPARE_REVIEWED=True`.
4. Khi đủ dữ liệu đã duyệt, bật `FINETUNE_ENABLED=True`. Giữ `BUILD_DRAFTS=False` khi train.
5. Section 3d: chấm baseline và từng checkpoint trên dev, chọn checkpoint bằng proxy;
   khóa lựa chọn rồi mới bật `RUN_HOLDOUT`. Chấm holdout một lần, chạy selection và merge.
6. Section 4: tự dùng output `merge-model` ở `artifacts/qwen15b_selected_v2` đã qua gate.
   Nếu model nằm nơi khác, đặt `FINETUNED_ARTIFACT` tới artifact đó. Muốn chạy baseline,
   đặt `FINETUNE_ENABLED=False` ở 3b và chạy lại cell chọn model ở section 4.

Notebook `notebooks/CASML_R1_inference_offline.ipynb` dành riêng cho inference Internet OFF.
Nó cần code v2, local model/tokenizer, retrieval cache và queries làm Input. Nếu thiếu dependencies,
chuẩn bị wheelhouse đúng môi trường trước; chỉ cài bằng `--no-index`. Kiểm tra ID/question của toàn
bộ cache trước generation. Kết quả được copy tới `/kaggle/working/submission.csv`.
Notebook phát triển có thể clone repo; nhớ dùng bản source đã chứa v2 qua `ROOT_OVERRIDE`,
hoặc cập nhật repo trước. Các sửa đổi local không tự xuất hiện trên GitHub/Kaggle.

## 1. Tạo bản nháp và đóng gói context

Các ví dụ lệnh chạy từ project root, dùng đường dẫn artifact của bạn. Luôn dùng output mới
khi đổi config/code/input. Cần Python 3.10–3.12 và dependencies trong `requirements-finetune.txt`
cho training; một CUDA GPU hiển thị. Không dùng API sinh nhãn bên ngoài.

```bash
python -m pip install -r requirements-finetune.txt
python -m casml_b0 build-sft --corpus artifacts/corpus_printed_v3 --config configs/synthetic.yaml --out artifacts/drafts_v2
python -m casml_b0 retrieve --index artifacts/index_bge_v3 --queries artifacts/drafts_v2/questions.json --config configs/retrieve.yaml --out artifacts/draft_retrieval_v2
python -m casml_b0 attach-review-context --drafts artifacts/drafts_v2/drafts.jsonl --retrieval artifacts/draft_retrieval_v2 --config configs/generate.yaml --out artifacts/context_drafts_v2
```

`build-sft` chỉ xuất `drafts.jsonl`, `questions.json`, report và checkpoint từng lần sinh;
không xuất `train.jsonl`. Answer không cần là substring nguyên văn; **quote hỗ trợ** phải có
nguyên văn trong corpus. Quote đúng không chứng minh claim/câu hỏi đúng. Tất cả nhãn mới là
`review_status: pending`. Đáp án bị cắt hoặc JSON sai bị loại.

Chia theo section nguồn trước khi sampling (fallback block 5 trang nếu thiếu section).
Sau retrieval, nguồn có thể vượt split ban đầu: người duyệt phải gom lại topic/source groups,
loại hoặc phân lại mẫu và khóa split. `prepare-sft` chặn cùng group, source section, trang,
text chunk hoặc required fact chuẩn hóa xuất hiện ở nhiều split. Câu hỏi gần trùng về ngữ nghĩa
vẫn cần người kiểm tra. Builder một chunk chỉ là bản nháp; bổ sung thủ công câu so sánh/multi-hop,
liệt kê và attribution có đủ nguồn, không coi số lượng draft là độ bao phủ đã đạt.

## 2. Annotation và duyệt độc lập

File annotation là JSONL. Mỗi hàng dùng các trường sau; ID nguồn phải tồn tại trong corpus:

```json
{
  "query_id": "book_001",
  "source_corpus_id": "ARTIFACT_ID_FROM_CORPUS_MANIFEST",
  "question": "A new question written from the book?",
  "answer": "A concise source-supported answer.",
  "question_type": "definition",
  "split": "train",
  "split_group": "topic-family-001",
  "evidence_ids": ["ACTUAL_CHUNK_ID"],
  "required_facts": [{"fact_id": "f1", "text": "One fact needed to answer the question.", "supports": [{"chunk_id": "ACTUAL_CHUNK_ID", "quote": "EXACT QUOTE FROM CORPUS"}]}],
  "answer_claims": [{"text": "A concise source-supported answer.", "supports": [{"chunk_id": "ACTUAL_CHUNK_ID", "quote": "EXACT QUOTE FROM CORPUS"}]}],
  "reference_sets": [["ACTUAL_CHUNK_ID"]],
  "review_status": "pending",
  "reviewer": "",
  "review_sha256": null
}
```

Ví dụ là cấu trúc minh họa, **không phải mẫu đã duyệt**; khi lưu JSONL viết mỗi object trên một dòng.
Reviewer phải kiểm tra câu hỏi, đủ ý, các claim nguyên tử theo thứ tự answer, quan hệ chủ thể,
phủ định, năm/số, quote, số trang in và section. Các `answer_claims.text` ghép lại phải phủ hết answer;
nguồn của answer phải nằm trong context thực tế (`evidence_ids`, theo đúng thứ tự prompt).
`required_facts` ghi đủ ý dù retrieved context thiếu nguồn: thiếu coverage không được xóa fact để tăng điểm.
Mỗi `reference_sets` là một tập nguồn thay thế đầy đủ, có hỗ trợ mọi fact; không trộn trang của phương án A
với section của phương án B. Context và references được tái dựng từ corpus, không tin text copy sửa tay.

Sau khi một hàng thật sự được đọc và duyệt, reviewer đặt `review_status="approved"`, tên/ID `reviewer`,
và tính `review_sha256 = review_digest(row)` bằng `casml_b0.reviewed.review_digest`.
Đây là dấu kiểm tra phiên bản nội dung, **không phải hệ thống xác thực người duyệt**. Sửa answer,
facts, nguồn hoặc split phải duyệt và tính lại digest. `source_corpus_id` phải khớp artifact corpus đã duyệt;
đổi corpus/page map phải review lại. Không đặt approved hàng loạt để vượt gate.
Gold dev/holdout cần được viết/duyệt độc lập trước khi xem dự đoán của model. Giữ nguồn và giấy phép khi chia sẻ.

Audit độc lập một mẫu phân tầng theo loại câu hỏi như kế hoạch v2 (đúng/đủ ≥95%, không sai nguồn nghiêm trọng)
là bước con người phải hoàn thành trước train. Code kiểm tra cấu trúc/nguồn/split, không thay việc audit ngữ nghĩa này.

```bash
python -m casml_b0 prepare-sft --corpus artifacts/corpus_printed_v3 --annotations annotations/reviewed.jsonl --config configs/reviewed.yaml --out artifacts/reviewed_v2
```

Mặc định cần ≥200 train, ≥80 dev, ≥80 holdout approved. Output `reviewed_sft` chứa ba file JSONL,
`split_manifest.json`, hashes và metadata audit; bản chưa duyệt bị bỏ qua, bản approved lỗi bị từ chối.
Có thể dùng cấu hình nhỏ cho kiểm thử, nhưng không hạ gate production sau khi nhìn kết quả.
Nguồn section/page hiện tại còn phụ thuộc corpus đã prepare; kiểm tra kiểu dữ liệu không chứng minh mapping đúng.
Sửa page map/canonical sections là thí nghiệm riêng, cần corpus và dataset version mới.

## 3. Train và lịch checkpoint

```bash
python -m casml_b0 finetune --dataset artifacts/reviewed_v2 --config configs/finetune.yaml --out artifacts/qwen_candidates_v2
```

Mặc định LoRA r8/alpha16/dropout0.05, Q/K/V/O, LR1e-5, cosine, warmup5%, một epoch,
batch1 × accumulation8 trên một GPU. BF16 nếu hỗ trợ, nếu không FP16. `max_length=3072` dành chỗ
cho context1900 token, prompt và answer; mẫu vượt trần bị từ chối, không cắt âm thầm. Chỉ answer
và token kết thúc lượt được tính loss; prompt/padding được mask.

Vòng train tính loss theo số answer token, xử lý cả nhóm accumulation cuối, xác nhận đủ mẫu và
optimizer steps. 41 mẫu × batch1/accumulation8 phải có **6** steps/epoch; 800 mẫu là100 steps.
Log LR, epoch thực, examples_seen trong `trainer_state.json` và `training_log.jsonl`.
Base step0, mỗi25 updates và cuối epoch đều sinh dev answers thật. Các thư mục:

- `checkpoints/step-000025`, `checkpoints/epoch-001`: adapter và tokenizer.
- `dev_generations/base`, `dev_generations/<checkpoint>`: predictions, review template, manifest.
- `report.json`: lịch train, môi trường, danh sách candidates, `selection_status: not_evaluated`.

Không đọc nhãn holdout để tối ưu, không chọn checkpoint theo `eval_loss`, không tự merge cuối train.
Với FP16/BF16, train/dev và inference có thể chịu sai số số học; xác nhận ứng viên bằng holdout tại
runtime inference thực và seed train thứ hai như kế hoạch. Chưa có checkpoint/gold thật được chạy trong repo này.

## 4. Chấm proxy và A/B

Copy `review_template.jsonl` ra ngoài artifact; reviewer chấm output ẩn tên model. Điền:

```json
{"query_id":"book_001","prediction_sha256":"KEEP_HASH_FROM_TEMPLATE","review_status":"approved","reviewer":"REVIEWER_ID","context_relevant_spans":[[0,42]],"claims":[{"text":"EXACT GENERATED CLAIM TEXT","supported":true,"correct":true,"fact_ids":["f1"]}],"severe_error":false}
```

`context_relevant_spans` là khoảng ký tự Python `[start,end)` trên **context đầy đủ** (kể cả nhãn E/page).
Giữ annotation relevance giống nhau cho baseline/candidate cùng context. `claims.text` ghép lại phải phủ
hết answer theo đúng thứ tự. Tách mệnh đề nguyên tử theo một rubric cố định; không bỏ câu sai hay tạo
claim rỗng. `supported` kiểm tra context thực tế; `correct` kiểm tra gold và nguồn, có thể khác supported.
`fact_ids` chỉ chứa ý bắt buộc được claim đúng đáp ứng. Ý đúng bổ sung có thể có `fact_ids: []`.
`severe_error` phải được quyết định rõ ràng; không suy ra từ việc có tên/năm trong context.

Spec `casml-local-proxy/v2-human-claims`:

| Chỉ số | Cách tính |
|---|---|
| CP | Token whitespace được span liên quan chứa trọn / tất cả token context; overlap chỉ đếm một lần |
| AF | Claim được actual context hỗ trợ / tổng claim được reviewer tách |
| AC | F1 của correct-claim precision và required-fact recall |
| RA | Max trên các bộ nguồn hợp lệ của trung bình page-set F1 và section-set F1, cùng một phương án |
| S_proxy | 0.20 CP + 0.20 AF + 0.40 AC + 0.20 RA, macro-average theo câu |

Thiếu review/facts, answer rỗng hoặc review sai prediction hash sẽ lỗi, không tự gán điểm0/1.
Precision=recall=0 cho F1=0; gold references rỗng không hợp lệ. `evidence_coverage` báo tỷ lệ
required facts có ít nhất một source chunk trong context, chỉ là kiểm tra source-ID, không kiểm tra entailment.
Không dùng S_proxy như điểm CASML chính thức: `official_metric_verified=false`.

```bash
python -m casml_b0 score-eval --predictions artifacts/qwen_candidates_v2/dev_generations/base --reviews annotations/dev_base.jsonl --config configs/evaluation.yaml --out eval/dev_base
python -m casml_b0 score-eval --predictions artifacts/qwen_candidates_v2/dev_generations/epoch-001 --reviews annotations/dev_candidate.jsonl --config configs/evaluation.yaml --out eval/dev_candidate
python -m casml_b0 select-model --baseline eval/dev_base --candidate eval/dev_candidate --config configs/evaluation.yaml --out eval/dev_decision
```

Chấm tất cả candidates trên dev, chọn S_proxy tốt nhất trong nhóm qua gate. Dev-only có trạng thái
`insufficient_evidence` vì chưa có holdout; đọc `gates` để quyết định có nên mở holdout cho ứng viên đã khóa.
A/B chặn khác dataset, base model/revision, prompt/context/reference/decoding fingerprint và CP/RA.
Kiểm tra thêm câu trả lời theo loại câu hỏi; không tối ưu chỉ macro-average.

Để phân biệt lỗi generator với retrieval, chạy thêm oracle context từ bộ nguồn đầy đủ đầu tiên đã duyệt:

```bash
python -m casml_b0 evaluate --dataset artifacts/reviewed_v2 --split dev --context-mode oracle --config configs/generate.yaml --out eval/oracle_base
python -m casml_b0 evaluate --dataset artifacts/reviewed_v2 --split dev --context-mode oracle --training artifacts/qwen_candidates_v2 --checkpoint epoch-001 --config configs/generate.yaml --out eval/oracle_candidate
```

Chấm riêng hai output oracle bằng `score-eval`; oracle chỉ chẩn đoán, không dùng để promote model.
Generated dev trong train mặc định là retrieved context.

## 5. Holdout, selection, merge

Chỉ chạy các lệnh sau sau khi khóa checkpoint bằng dev; không dùng holdout để thử nhiều checkpoint.

```bash
python -m casml_b0 evaluate --dataset artifacts/reviewed_v2 --split holdout --config configs/generate.yaml --out eval/holdout_base_predictions
python -m casml_b0 evaluate --dataset artifacts/reviewed_v2 --split holdout --training artifacts/qwen_candidates_v2 --checkpoint epoch-001 --config configs/generate.yaml --out eval/holdout_candidate_predictions
python -m casml_b0 score-eval --predictions eval/holdout_base_predictions --reviews annotations/holdout_base.jsonl --config configs/evaluation.yaml --out eval/holdout_base
python -m casml_b0 score-eval --predictions eval/holdout_candidate_predictions --reviews annotations/holdout_candidate.jsonl --config configs/evaluation.yaml --out eval/holdout_candidate
python -m casml_b0 select-model --baseline eval/dev_base --candidate eval/dev_candidate --holdout-baseline eval/holdout_base --holdout-candidate eval/holdout_candidate --config configs/evaluation.yaml --out eval/confirmed_decision
python -m casml_b0 merge-model --training artifacts/qwen_candidates_v2 --checkpoint epoch-001 --selection eval/confirmed_decision --config configs/evaluation.yaml --out artifacts/qwen_selected_v2
```

Dùng chính config generation runtime của lần train khi làm notebook A/B, không đổi prompt về config
mặc định giữa chừng. Gate mặc định: ≥80 dev, ≥80 holdout, ≥20 dev groups; dev S_proxy tăng≥0.02,
paired group bootstrap2000 lần/seed42 có cận dưới CI95%>0; AF/AC không giảm; không lỗi nghiêm trọng
mới; schema/EOS đạt. Holdout phải cùng exact checkpoint/base/dataset, proxy tăng và AF/AC không giảm.
Mẫu quá ít hoặc thiếu holdout → `insufficient_evidence`; đủ mẫu nhưng fail gate → `rejected`.
Cả hai giữ baseline. Threshold là policy dự án, không phải ngưỡng Kaggle. `bootstrap_group` được
prepare-sft tạo theo nhóm liên thông của topic/fact/section/page/chunk, để paraphrase cùng nguồn không
bị tính thành mẫu độc lập dù người gán nhãn đặt `split_group` khác nhau.

`merge-model` từ chối checkpoint không được chọn. So sánh logits trên ba prompt dev cố định trước/sau
merge bằng CPU FP32 (`atol=rtol=1e-4`), rồi lưu và tải lại model/tokenizer để kiểm tra lần nữa.
Chỉ sau đó mới hoàn tất artifact, tạo `generate.yaml`, `selection.json`, report và checksums.
Nếu chuyển artifact sang Kaggle Input, helper `selected_generation_config` sửa đường dẫn lúc load;
không sửa các file bên trong artifact. Logit check không thay kiểm tra toàn bộ inference offline.

## Ngắt giữa chừng và phạm vi kiểm chứng

- `build-sft`/`evaluate` lưu checkpoint từng câu và resume cùng code/config/input; checksum sai sẽ dừng.
- Train, prepare, attach-context và merge **chưa resume giữa chừng**. Nếu bị ngắt, artifact còn incomplete,
  không được chọn cho inference; giữ nguyên để audit và dùng output mới khi chạy lại. Không sửa complete=true.
- Không chỉnh annotation/review bên trong artifact đã finish; copy ra file riêng, xuất artifact version mới.
- Unit/integration tests dùng fixture nhỏ và mock generator; không chứng minh Qwen train/merge GPU thành công
  hoặc điểm private sẽ tăng. Cần dữ liệu con người duyệt, CUDA pilot, seed xác nhận và chạy offline thật.
- Nhánh sửa retrieval nhiều vế/canonical section mapping P2 cần thí nghiệm riêng; phiên bản này giữ baseline
  retrieval để đo riêng fine-tune. Không giả định đã sửa ngữ nghĩa page/section của sách.
