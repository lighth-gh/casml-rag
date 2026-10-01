# Review CASML R1: Version 11 → Version 13

## Kết luận sau đối chiếu trực tiếp

Đã khoanh vùng chính xác thay đổi submission: cả 50 context và references giống hệt; 33 answer giống hệt; chỉ 17 answer bị grounding retry/repair thay đổi. Vì vậy chênh lệch điểm giữa hai submission nằm ở các answer bị bước grounding mới sửa, không phải do thay model, prompt, context hoặc references. Không có metric/gold từng câu nên không thể chia mức giảm điểm cho từng answer hoặc kết luận cả 17 câu đều kém hơn.

Ảnh leaderboard người dùng cung cấp: public 0.317 → 0.295 (−0.022, −6.94%); private 0.334 → 0.320 (−0.014, −4.19%).

## Artifacts và provenance

| Trường | Version 11 | Version 13 |
|---|---|---|
| Diagnosis | C:/Users/HP/Downloads/diagnosis11.json | C:/Users/HP/Downloads/diagnosis.json |
| Submission | C:/Users/HP/Downloads/submission11.csv | C:/Users/HP/Downloads/submission.csv |
| Log | casml-r1 (4).log | casml-r1 (3).log |
| Generation ID | 2acc259830a4149d51174382346c4140278f86dcfd38364cb6b4704518cafa0c | 60718c4c0ce4c24b4df57ec0f4226dcd73e4acda231e8ca109e90216f125bcbe |
| Retrieval ID | e108971be4a279fd9168c6b3420f4e23e31ac817babfac19bdb0aa16516d92c5 | e108971be4a279fd9168c6b3420f4e23e31ac817babfac19bdb0aa16516d92c5 |
| Model/revision | Qwen/Qwen2.5-1.5B-Instruct, 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | Giống bản 11 |
| Initial decoding | 384 tokens, greedy, repetition penalty 1.05, no-repeat-ngram 6, seed 42 | Giống bản 11 |
| Prompt/packing | 140-word system prompt, top 4, budget 1900 | Giống bản 11 |
| Grounding | Chưa có | Retry 1 lần; 224 tokens; penalty 1.15; no-repeat-ngram 8; deterministic repair |

Hash generation.py của bản 11 khớp blob 1d9c91a, bản 13 khớp blob fd3d2b8. Điều này định vị thay đổi ở fbaeaeb (thêm validator) và fd3d2b8 (sửa retry + thêm fallback). Signature không chứa đầy đủ Git commit, nên hash module xác nhận phiên bản module chứ không tự chứng minh mọi file trong checkout thuộc đúng commit đó.

Environment Python/packages giống hệt. Diff 1d9c91a..HEAD không đổi retrieval.py, context.py, llm.py, prompt hoặc configs prepare/index/retrieve.

## Kết quả kiểm tra

- Hai CSV đều có 50 rows, cùng IDs/thứ tự; answer/context khớp diagnosis tương ứng.
- Context, references và questions giống hệt 50/50 câu.
- Selected evidence có cùng text/chunk/source/page/rank và các metadata khác; chỉ 51 giá trị bm25_score khác ở sai số tối đa 0.00000190735. Selected-query signature khác, nhưng không có khác biệt context đưa vào model. Không đủ raw retrieval rows để kết luận nguyên nhân mọi khác biệt digest.
- Initial messages giống 50/50 câu. Initial attempt token counts/settings, output token count và answer head/tail khớp 50/50. Không lưu full initial draft ở mọi retry, nên không tuyên bố đã so khớp toàn bộ từng byte của 50 initial answers. 9 full original answers được lưu trước repair khớp chính xác bản 11.
- Các answer thay đổi trùng đúng danh sách 17 grounding retries; 33 answer còn lại không đổi.
- 12 deterministic repairs đã replay chính xác từ full source draft và evidence.
- Không có error, cutoff hoặc length retry ở cả hai run. retry_instruction 180 → 140 words không phải đường chạy tạo khác biệt ở hai run này, vì không có length retry.

## Findings theo mức ưu tiên

### High: fallback xóa lỗi kèm cả nội dung trả lời chính

validation.py:160–213 xóa parenthetical hoặc whole sentence chứa token bị flag; giữ prefix dưới 140 từ. generation.py:126–142 chọn candidate theo số câu/fragment bị xóa rồi ưu tiên số từ nhỏ hơn, không đánh giá đúng ý hoặc completeness.

| Query | Bản 11 | Bản 13 | Tác động quan sát |
|---|---:|---:|---|
| 11: history of psychology | 129 từ | 53 từ | Mất đoạn mở đầu và structuralism; sai liên hệ humanistic/unconscious vẫn tồn tại. |
| 30: psychological disorders | 95 từ | 38 từ | Xóa phần định nghĩa, chỉ còn distress/impairment và ý chung về treatment. |
| 31: DSM-5 | 71 từ | 17 từ | Xóa cả tên đầy đủ và các chi tiết; còn duy nhất một câu bắt đầu bằng “It”. |
| 50: critical thinking | 216 từ | 132 từ | Bỏ phần ứng dụng cụ thể và kết luận; retained prefix còn lặp ý. |

Q31 bản 11 có lỗi thật: 568 disorders, 22013 và cụm “published in 237 specific diagnoses”. Xóa các lỗi này có ích, nhưng fallback bỏ luôn thông tin cần thiết thay vì dùng nguồn để sửa. Evidence đã có 2013, 237 diagnosable disorders và diagnostic criteria/risk factors. Vì vậy rollback phục hồi điểm baseline nhưng không đồng nghĩa bản 11 là factual ground truth.

Q50 có phần cuối bị cắt vì word limit nhưng removed_sentences=[]; metadata không liệt kê toàn bộ nội dung mất đi.

### High: retry thêm claim thiếu hỗ trợ

Q48 bản 13 thêm so sánh “personality disorders ... less likely ... depression or bipolar disorder” và ý về “higher rates ... hospitalizations”. Bản 11 không có các claim này. Context của cả hai chỉ chứa personality-disorder material, trong đó trang 584 nêu borderline comorbid với mood disorders và trang 583–584 nêu instability/suicidal behavior. Nguồn không đưa so sánh tỷ lệ như retry khẳng định. Đây là lỗi mới sau rewrite; validator vẫn trả valid=True.

Q12 retry tạo chuỗi dính chữ “WilhelmWundtwasacredtoforscientificpsychologyasadistinctdiscipline.”; word_count=2. Fallback quay lại draft bản 11 và xóa vài câu, không sửa quan hệ Wundt/James/Freud.

### Medium: proper-name heuristic false positives và token support quá yếu

Q30 retry flag “Clinically Significant Disturbances” vì cách viết hoa và dạng từ, dù evidence có “clinically significant disturbance”. Replay một định nghĩa viết thường pass; cùng ý viết hoa bị flag, rồi xóa cả định nghĩa.

Q31 tên đầy đủ DSM bị flag do thiếu literal trong 4 chunk được chọn. Thiếu literal không chứng minh thuật ngữ sai; cần xử lý evidence coverage hoặc viết lại có nguồn thay vì xóa câu tự động.

Q12 bản 11 gán functionalism cho James đúng, nhưng đã gán dream analysis/slips of tongue của Freud cho James. Bản 13 xóa câu giới thiệu James rồi nối các câu còn lại; “He was a proponent of the functionalist perspective” lúc này nằm ngay sau các câu về Wundt. Đây là lỗi antecedent do repair mới tạo ra. Evidence nói Wundt structuralist, James functionalist, Freud dùng dream analysis. Q26 cả hai gán postformal thought cho Piaget trong khi evidence nói những người khác đề xuất stage này và bất đồng với Piaget. Lỗi dream analysis và postformal attribution đã có ở bản 11; push mới giữ lại chúng. Lỗi functionalism antecedent ở Q12 là regression sau khi xóa câu. Validator token support không phát hiện quan hệ sai.

### Medium: word-count check kích hoạt rewrite không cần grounding

Q14, Q36, Q42, Q50 retry chỉ vì quá 140 từ; initial không có unsupported numeric/name tokens. Q36 202 → 54 từ, Q14 162 → 85 từ. Đây là giới hạn mới được cưỡng chế bởi validator (system prompt trước đó đã yêu cầu 140 nhưng model không luôn tuân thủ).

Log ghi “Unsupported evidence tokens detected” cho cả các trường hợp length-only, dễ hiểu sai nguyên nhân.

### Có cải thiện ở một số câu

Không phải mọi retry đều xấu hơn. Q38 sửa “Elektro” thành “Electra” và giữ đủ 5 stages, dù thêm marker [E1]. Q41 chỉ loại acronym SSRIs khỏi parenthetical; nhiều nội dung khác giữ nguyên. Q45 loại tên “Prossers” và một số chi tiết ngoài các excerpt. Chưa có evaluator từng câu để kết luận lợi/hại tuyệt đối cho nhóm này.

## Toàn bộ 17 answer thay đổi

| ID | Số từ bản 11 | Số từ bản 13 | Final finish |
|---|---:|---:|---|
| 11 | 129 | 53 | grounding_repair |
| 12 | 187 | 129 | grounding_repair |
| 13 | 97 | 83 | grounding_repair |
| 14 | 162 | 85 | eos |
| 26 | 89 | 83 | grounding_repair |
| 27 | 73 | 57 | grounding_repair |
| 29 | 95 | 56 | grounding_repair |
| 30 | 95 | 38 | grounding_repair |
| 31 | 71 | 17 | grounding_repair |
| 33 | 123 | 93 | grounding_repair |
| 36 | 202 | 54 | eos |
| 38 | 93 | 135 | eos |
| 41 | 129 | 128 | grounding_repair |
| 42 | 173 | 103 | grounding_repair |
| 45 | 144 | 101 | eos |
| 48 | 213 | 132 | eos |
| 50 | 216 | 132 | grounding_repair |

Tổng trên 17 câu: 2291 → 1479 từ (−35.44%). Đây là dấu hiệu mức sửa mạnh, không phải thước đo chất lượng. Full side-by-side answers và validations ở r11-r13-comparison.json.

## Đề xuất xử lý

1. Giữ submission11.csv làm baseline đã biết điểm. Khi cần khôi phục hành vi, quay về policy generation trước grounding; giữ các fixes EOS/retrieval/export trước đó.
2. Thử cấu hình hiện tại với grounding_validator_enabled=false và fail_on_unsupported_claims=false ở exporter, dùng run/output mới trên cùng retrieval cache. Hai flags phải đi cùng để export gate mới không chặn baseline. Giữ nguyên model/revision/prompt/seed/initial decoding. Chưa thực hiện thay đổi này trong review.
3. Đưa heuristic về chế độ report-only trước khi có validation quality. Ngừng xóa whole sentences để ép valid; sửa claim theo nguồn và kiểm tra giữ được định nghĩa/ý chính.
4. Tách length checks khỏi grounding. Đánh giá retry và repair riêng trên nhóm 17 câu; ghi full draft và mọi đoạn bị bỏ. Không gộp đổi chat history, token budget và decoding vào một cải tiến chưa ablate.
5. Các lỗi nền Q12/Q26 và thiếu mood context Q48 xử lý riêng, dùng bản 11 để đối chiếu nhưng không làm gold answer.

Không thể tính lại hidden leaderboard score hoặc điểm từng query từ các files này. Tuy nhiên khác biệt của hai CSV đã được khoanh vùng hoàn toàn vào 17 answer grounding sửa.

Review chỉ tạo/cập nhật docs/reviews; không sửa production config/code/notebook và không push.
