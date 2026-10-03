# Kế hoạch fine-tune Qwen 1.5B theo tiêu chí CASML sau lần chạy đạt 0.330

Ngày lập: 02/10/2026; cập nhật triển khai 03/10/2026, Asia/Saigon. Phiên bản **v2**.
Đã triển khai workflow code/notebook cho draft → reviewed dataset → LoRA candidates → đánh giá → selection → merge và inference offline.
**Chưa tạo gold đã duyệt hoặc chạy Qwen/GPU pilot mới; chưa có bằng chứng cải thiện private score.**
Xem lệnh, schema thực tế và giới hạn trong [FINETUNING.md](FINETUNING.md).
Các checklist kết quả thí nghiệm dưới đây vẫn cần dữ liệu và lần chạy thật; không tự đánh dấu đạt chỉ vì code đã có.
Nhánh P2 sửa retrieval nhiều vế/metadata, audit ngữ nghĩa ≥95%, xác nhận seed thứ hai và kiểm tra Kaggle offline thật còn cần thực hiện riêng.

Mục tiêu là cải thiện chất lượng câu trả lời có nguồn trên bộ đánh giá độc lập, với đủ bốn thành phần CASML, trước khi thay baseline bằng model fine-tune. Không tiếp tục lấy loss trên đáp án sao chép làm tiêu chí quyết định. Điểm private **0.330 là thông tin người dùng cung cấp**; các file hiện có không chứa bảng điểm Kaggle hay breakdown của metric.

### Quyết định thực hiện trong bản v2

1. **Khóa baseline trước.** Khi triển khai, tắt tự động sinh nhãn/train/chọn model trong luồng submission. Tách ba hành động: tạo dữ liệu, huấn luyện thử và chọn model để inference.
2. **Đổi cách tạo nhãn.** Tạo câu hỏi từ các ý đã xác định trong sách; cho phép answer diễn đạt lại, nhưng cần ánh xạ từng mệnh đề tới nguồn và kiểm tra câu hỏi–đáp án.
3. **Đo riêng generator.** Fine-tune mới phải được so với Qwen gốc trên cùng input và reference policy. Sửa retrieval là nhánh thí nghiệm riêng, không bắt mọi sửa đổi retrieval phải xong mới thử SFT.
4. **Chọn bằng output sinh thực.** `eval_loss` chỉ theo dõi tối ưu; chọn checkpoint trên dev đã review theo AF/AC và điểm proxy đủ bốn thành phần, rồi xác nhận trên holdout.
5. **Giữ baseline khi chưa có bằng chứng tốt hơn.** File model tạo thành công hoặc bộ kiểm thử code đạt không cho phép tự đổi model submission.

Phạm vi thực hiện sau bản kế hoạch: Qwen2.5-1.5B-Instruct + LoRA, schema submission hiện có và các artifact tách biệt. Chưa mở thêm DPO/RL, thay model lớn hoặc sửa tay đáp án của 50 câu test để làm nhãn train.

## 1. Kết luận từ lần chạy hiện tại

**Đã fine-tune thật và đã dùng model đã fine-tune để tạo submission.** Đây không còn là trường hợp section 3b bị bỏ qua.

| Kiểm tra | Bằng chứng | Ý nghĩa |
|---|---|---|
| Model inference | `.../artifacts/qwen15b_book_lora_v2/model` trong `diagnosis.json` | Đã load model local sau train |
| Artifact fine-tune | `590f60fd20925a1cc8bdf5f93a82abbeabde75c1820084cf20696bb448900b5d` | Log và diagnosis khớp nhau |
| Dữ liệu thực tế | 41 train, 7 validation | Không đạt mục tiêu cấu hình 180/20; pipeline vẫn cho train vì ngưỡng tối thiểu chỉ 20 cặp tổng cộng |
| Tỷ lệ giữ dữ liệu | 48/600 = 8%; loại 552 cặp | Bộ train rất nhỏ và bị bộ lọc chọn thiên về khả năng sao chép |
| Lý do loại | 500 `answer_not_exact_source_span`, 46 `answer_length`, 6 lý do khác | Không được diễn giải tất cả 500 trường hợp là câu trả lời sai nghĩa; chưa có các cặp gốc để đánh giá điều đó |
| Tham số LoRA được train | 4.358.144 trên 1.548.072.448 tham số | Có cập nhật adapter; không phải chỉ đổi prompt |
| Training progress | Thanh tiến trình `2/2`, báo cáo epoch ≈ 0,762; train runtime ≈ 48,06 giây | Chỉ hai optimizer steps được ghi nhận; cần kiểm tra batch, số GPU và gradient accumulation |
| Loss validation trước/sau | 0,199103 → 0,183989 trên 7 mẫu | Chỉ chứng minh dự đoán token của các nhãn này tốt hơn; chưa chứng minh câu trả lời sinh tự do tốt hơn |
| Hoàn tất generation | 50/50, không error, không length cutoff | Đạt yêu cầu chạy; không đồng nghĩa đúng nội dung |
| Cấu trúc CSV | 50 ID duy nhất có dạng số nguyên; đúng `ID,context,answer,references`; không ô rỗng | Không phát hiện lỗi cấu trúc cơ bản |
| Đối chiếu CSV với diagnosis | 50 answer/context khớp; tập pages/sections khớp metadata evidence | Không thấy xuất nhầm đáp án; thứ tự pages đã sắp xếp không phải lỗi |
| Cảnh báo nội dung | 13/50 câu; 5 câu vượt ngưỡng nội bộ 140 từ | Bộ kiểm tra hiện tại chỉ bắt được một số vấn đề |
| Metadata nguồn | 200/200 lượt evidence dùng section gần đúng theo TOC và trang in từ offset | Chưa có xác minh ngữ nghĩa section/page ở cấp đoạn |

Không có đủ bằng chứng để kết luận riêng LoRA gây ra toàn bộ mức giảm điểm. Thiếu submission baseline tương ứng, điểm baseline được xác minh và breakdown private. Cần chạy so sánh có kiểm soát, không mặc định lỗi do learning rate, overfitting hay page offset.

### Các lỗi cụ thể có thể kiểm tra ngay

| Query ID | Quan sát trong file | Công việc cần ưu tiên |
|---|---|---|
| 11 | Đáp án ghi Wundt xuất bản năm **1874**, evidence ghi **1873**. Đáp án còn gắn unconscious/early childhood với humanistic psychology, trong khi evidence gắn chúng với psychoanalytic theory | Kiểm tra từng mệnh đề và quan hệ chủ thể–thuộc tính; sự hiện diện của từ khóa không đủ |
| 2 | Đáp án viết soma chứa cả nucleus và dendrites; nguồn phân biệt nucleus nằm trong soma, dendrites là các nhánh mở rộng từ soma. Context còn có đoạn về somatic/autonomic nervous system | Nhãn phải giữ đúng quan hệ giải phẫu được mô tả; giảm đoạn gần chủ đề nhưng không trả lời câu hỏi |
| 48 | Câu hỏi so sánh mood disorders với personality disorders, nhưng cả bốn chunk thuộc `15.11 Personality Disorders`; đáp án vẫn suy rộng thành sáu tiêu chí, dài 213 từ | Retrieval phải tìm đủ hai vế; không dùng fine-tune để bù evidence thiếu bằng kiến thức suy đoán |
| 42, 47, 50 | Context có đoạn Review/Personal Application/Critical Thinking Questions | Đánh giá đóng góp thật của các đoạn này; không tự coi trang bài tập là đáp án |

Chín câu có ít nhất một đoạn thuộc nhóm bài tập/câu hỏi: 13, 16, 21, 26, 28, 41, 42, 47, 50. Đây là dấu hiệu cần audit, chưa phải kết luận tất cả chín câu đều sai.

Các ID vượt 140 từ: 12, 14, 42, 45, 48. Độ dài trung bình 100,62 từ; median 96; khoảng 13–213 từ. **140 từ là policy của dự án, không phải giới hạn cứng được nêu trong mô tả cuộc thi.** Không xóa câu chỉ để đạt số từ nếu làm mất ý bắt buộc.

## 2. Metric chính thức: điều đã biết và điều chưa biết

Nội dung Overview người dùng cung cấp nêu các trọng số sau. Dataset Description trong tin nhắn yêu cầu đúng schema, đủ câu, context phù hợp và references kiểm chứng được.

| Thành phần | Trọng số công bố | Hành vi phải cải thiện | Phần pipeline chịu trách nhiệm |
|---|---:|---|---|
| Context Precision | 20% | Đoạn đưa vào context liên quan trực tiếp, ít nhiễu, đủ các vế câu hỏi | Retrieval, reranker, context packing |
| Answer Faithfulness | 20% | Mỗi mệnh đề được evidence hỗ trợ, không ghép nhầm quan hệ | Nhãn SFT, generator, kiểm tra mệnh đề |
| Answer Correctness | 40% | Đủ ý cần trả lời và đúng so với đáp án chuẩn | Chất lượng nhãn, coverage của retrieval, generator |
| Reference Accuracy | 20% | Mục và trang chính xác, đủ nguồn hỗ trợ đáp án | Metadata, chọn evidence, exporter |

**Chưa có mã scorer chính thức, phiên bản scorer, ground truth hoặc cách chuẩn hóa điểm.** Không được gọi RAGAS, cosine similarity, ROUGE, token overlap, loss hay công thức proxy tự viết là “metric CASML chính xác”. Bốn trọng số cũng không cho biết evaluator dùng judge nào, có phạt reference dư thế nào hay so section bằng chuỗi tuyệt đối hay ngữ nghĩa.

Việc đầu tiên là tìm tài nguyên “Sample code for evaluation metrics” được Overview nhắc đến. Nếu có, lưu URL/version/hash, đọc preprocessing, aggregation và đối chiếu sample. Nếu không có, tiếp tục với proxy minh bạch và giữ `official_metric_verified: false`; không chặn toàn bộ công việc chỉ vì thiếu scorer.

Không thể suy ra từ một private score duy nhất rằng thành phần nào đã giảm. Với context và references cố định, adapter generator chủ yếu thay đổi hai thành phần liên quan đến answer; cần thí nghiệm riêng để cải thiện hai thành phần còn lại.

## 3. Thay đổi định hướng so với thiết kế fine-tune trước

Luồng hiện tại tự sinh câu hỏi từ một chunk rồi yêu cầu đáp án nguyên văn. Bộ lọc xác minh được đoạn chữ có tồn tại nhưng không xác minh được câu hỏi có hợp lý, đáp án có đầy đủ hay các quan hệ có đúng. Inference lại dùng bốn chunk với câu hỏi so sánh, tổng hợp và context nhiễu. Đây là lệch giữa bài tập train và công việc cần thực hiện.

Kế hoạch thay bằng **SFT trên câu hỏi–evidence–đáp án đã kiểm tra**, có nhiều loại câu hỏi và context lấy qua đúng retrieval pipeline. Đáp án có thể diễn đạt lại, nhưng từng ý phải có nguồn. Trích đoạn nguyên văn được dùng làm bằng chứng gán nhãn, không phải điều kiện buộc toàn bộ answer phải là một substring.

- Không chỉ nới lỏng bộ lọc để lấy đủ số lượng.
- Không lấy 50 câu test hoặc paraphrase của chúng để tạo nhãn train.
- Không để cùng model vừa sinh nhãn vừa tự chấm rồi coi nhãn là gold.
- Không tăng epoch trên 41 cặp trước khi có đánh giá độc lập.
- Không tự bật model mới cho submission chỉ vì `finetune: complete=True`.

Các lỗi ở ID 2/11/48 dùng để hiểu loại lỗi và làm audit đọc kết quả, **không đưa nguyên câu/đáp án sửa vào train hoặc bộ đo cải thiện độc lập**.

## 4. P0 — Khóa baseline và đo lại tác động của LoRA

- [ ] Lưu nguyên run hiện tại, CSV, log, diagnosis, manifest và checksum. Gắn nhãn `private_score_reported=0.330`, không gắn nhãn model tốt nhất.
- [ ] Thu hồi bản baseline từng có điểm tốt hơn nếu còn. Ghi đúng public/private, thời điểm, submission hash; hiện chưa có giá trị baseline được xác minh.
- [ ] Tạo run A dùng Qwen2.5-1.5B-Instruct gốc và run B dùng adapter hiện tại trên **cùng câu hỏi, messages/context, decoding, code và exporter**.
- [ ] So sánh hash từng prompt/context và tập references. Cùng `retrieval_id` chưa đủ để khẳng định đóng gói context giống nhau.
- [ ] Với 50 câu cuộc thi, chỉ lập bảng khác biệt và audit lỗi. Không dùng điểm private lặp đi lặp lại để chỉnh nhãn hay tìm siêu tham số.
- [ ] Trong lần triển khai kế hoạch, chuyển cơ chế lựa chọn về baseline mặc định; model fine-tune chỉ được chọn sau khi qua tiêu chí ở mục 9.

Cụ thể ở notebook: `FINETUNE_ENABLED = False` và `FINETUNED_ARTIFACT = None` cho luồng baseline. Bật huấn luyện chỉ chạy thí nghiệm, không đồng thời cập nhật model đang được chọn. Code v2 đã thêm báo cáo lựa chọn ghi model, baseline so sánh, dataset/evaluator version và các gate; chưa có báo cáo thực nghiệm đạt gate.

Đầu ra dự kiến: `eval/baseline_vs_current.json`, bảng per-query gồm answer A/B, độ dài, nguồn, lỗi và các metric có thể tính. Nếu A/B không giữ nguyên đầu vào, ghi rõ yếu tố thay đổi trước khi diễn giải kết quả.

## 5. P1 — Xây bộ đánh giá độc lập và proxy theo bốn thành phần

### Dữ liệu đánh giá

Mốc ban đầu: **80 câu development và 80 câu holdout**, không lấy từ 50 câu test. Đây là cỡ mẫu đề xuất để bắt đầu, không phải bảo đảm độ tin cậy thống kê. Nếu ngân sách gán nhãn chưa đủ thì làm pilot nhỏ hơn, nhưng chưa cho phép thay model production.

1. Tạo câu hỏi mới từ sách, bao phủ định nghĩa, cơ chế, so sánh, liệt kê và tổng hợp.
2. Tách nhóm section/chủ đề trước khi tạo các paraphrase. Chunk chồng lấp, cùng source fact hoặc câu hỏi gần trùng phải cùng split. Chia theo trang như hiện tại chỉ là bước tối thiểu.
3. Người kiểm tra viết `required_facts`, đáp án tham khảo, source spans, section/page hỗ trợ. Reviewer không nhìn câu trả lời của model đang được chấm khi viết gold.
4. Ghi các phương án diễn đạt/nguồn thay thế hợp lệ; không coi duy nhất một câu chữ là đúng.
5. Holdout khóa lại; chỉ mở cho ứng viên cuối cùng. Không tiếp tục tối ưu trên holdout sau khi xem kết quả; nếu cần thì phải tạo holdout mới.

Schema nội bộ đề xuất cho mỗi mẫu: `example_id`, `question`, `question_type`, `split_group`, `required_facts`, `gold_answer`, `gold_evidence_spans`, `gold_pages`, `gold_sections`, `review_status`, `label_source`. Thêm `retrieved_context` và ánh xạ mệnh đề–evidence khi chạy đánh giá. Schema này **không thay đổi bốn cột submission**.

Tách hai lớp file: bản annotation đầy đủ phục vụ kiểm chứng; bản JSONL `query_id/question/context/answer` phục vụ trainer hiện tại. Bản export train phải lưu ID/hash liên kết về annotation. Các paraphrase hoặc phiên bản context của cùng câu hỏi giữ chung group và không được tính thành các mẫu độc lập khi bootstrap.

### Khi chưa có dữ liệu do người kiểm tra

Đây là đầu vào còn thiếu, không thể giải quyết bằng cách đổi tên dữ liệu synthetic thành gold. Có thể tự động tạo bản nháp và bảng review gồm câu hỏi, ý bắt buộc, đáp án, đoạn nguồn, trang/mục và lỗi nghi ngờ. Người kiểm tra xác nhận/sửa các trường rồi đặt `review_status=approved`.

- Nhãn tự sinh chưa kiểm tra: chỉ phục vụ kiểm thử pipeline hoặc thí nghiệm thăm dò; không dùng để chứng nhận model tốt hơn.
- Gold dev/holdout: phải được kiểm tra độc lập, có nguồn và trạng thái review; không lấy câu trả lời của model cần đánh giá làm chuẩn.
- Nếu nguồn lực chưa đủ 80 dev + 80 holdout: dùng tập nhỏ để phát hiện lỗi và kiểm tra quy trình, ghi rõ kết luận chưa đủ bằng chứng để thay baseline.
- Tự động hóa hỗ trợ reviewer và đóng gói dữ liệu; không giả định công việc review đã hoàn thành trong kế hoạch này.

### Proxy phải công khai cách tính

Tất cả tính per-query rồi lấy macro-average; không để câu dài chi phối toàn bộ tập.

| Proxy | Định nghĩa triển khai đề xuất | Kiểm soát giới hạn |
|---|---|---|
| CP | Tỷ lệ token thuộc các đoạn/span được gán nhãn hỗ trợ câu hỏi trong tổng context đưa vào model | Đếm cả phần nhiễu trong chunk; theo dõi thêm evidence recall/coverage để tránh tối ưu bằng context quá ít |
| AF | Số mệnh đề trong answer được context thực tế hỗ trợ chia tổng mệnh đề có thể kiểm chứng | Kiểm tra chủ thể, quan hệ, phủ định, số/năm; không chỉ tên riêng. Answer rỗng bị loại bởi schema gate |
| AC | Precision/recall/F1 của các ý đúng so với `required_facts` và ý bổ sung đã được nguồn xác nhận | Recall phạt thiếu ý; precision phạt ý sai. Ghi riêng lỗi mâu thuẫn nghiêm trọng; từ chối câu có đủ evidence phải giảm AC |
| RA | Trung bình F1 của tập trang và tập section so với các tập nguồn hỗ trợ hợp lệ được gán nhãn | Nếu nhiều tập nguồn đều hợp lệ, so với tập phù hợp nhất đã được reviewer xác nhận; không khớp máy móc một golden citation duy nhất |

Điểm tổng hợp phục vụ chọn thí nghiệm:

`S_proxy = 0.20 × CP + 0.20 × AF + 0.40 × AC + 0.20 × RA`

Đây là **proxy đề xuất theo trọng số công bố, không phải tái tạo scorer private**. Định nghĩa trường hợp mẫu số 0 trong spec trước khi chạy; không biến metric chưa gán nhãn thành 0 hoặc 1. Câu thiếu evidence vẫn chấm AC theo gold và ghi cờ retrieval thiếu; không “thưởng” từ chối toàn bộ để làm AF tăng.

Khóa evaluator trước khi so sánh: cùng tokenizer đếm context, cùng rubric tách mệnh đề, cùng quy tắc ghép ý, cùng mapping section và cùng xử lý thiếu dữ liệu. Với RA, chọn **một bộ nguồn thay thế hợp lệ duy nhất** để tính cả pages và sections; không chọn bộ trang tốt nhất từ annotation A rồi bộ section tốt nhất từ annotation B. Kiểm tra thêm tính nhất quán cặp section–page dù CSV chỉ xuất hai danh sách.

Khi so riêng generator và giữ context/references cố định, CP và RA phải bằng nhau; chênh lệch `S_proxy` lúc đó đến từ AF và AC. Nếu hai số này thay đổi ngoài dự kiến, dừng diễn giải thí nghiệm và kiểm tra input/evaluator. Nếu thử reference selection phụ thuộc answer, đánh dấu đó là thí nghiệm pipeline riêng.

Chấm bằng rubric nguồn và review ẩn tên model. Công cụ tự động có thể gợi ý nhãn, nhưng một mẫu gán nhãn tay cố định phải được dùng để đo mức đồng thuận. LLM judge chạy local, model công khai ≤3B nếu dùng; phải hiệu chuẩn bằng review tay và công bố độ bất đồng. Không dùng API bên ngoài trong inference submission.

Hai chế độ đánh giá bắt buộc: **oracle context** để đo khả năng generator khi đủ nguồn; **retrieved context** để đo hệ thống end-to-end. Điểm oracle tốt nhưng end-to-end kém là tín hiệu sửa retrieval, không phải cứ train generator thêm.

## 6. P2 — Sửa evidence và references trước khi mở rộng train

P2 là nhánh sửa pipeline được đo bằng Qwen gốc trước. Có thể làm song song về tiến độ với chuẩn bị dữ liệu P3, nhưng không trộn kết quả hai nhánh vào cùng một phép so sánh. SFT pilot có thể dùng policy retrieval đã khóa hiện tại nếu context của từng mẫu train được kiểm tra đủ nguồn; chỉ các mẫu thiếu evidence mới cần sửa trước khi đưa vào train.

- [ ] Với câu hỏi so sánh A/B, truy hồi cho từng vế rồi hợp nhất/rerank; xác nhận context có evidence cho cả hai. Kiểm tra loại lỗi ID 48 trên câu mới trong dev.
- [ ] Với câu hỏi lịch sử/quy trình/liệt kê, phân rã các ý cần bao phủ và chọn context theo coverage, tránh bốn chunk cùng một chi tiết.
- [ ] Audit nhóm Review Questions/Personal Application/TOC/header. Loại hoặc giảm ưu tiên khi chỉ nêu câu hỏi; giữ các phần Summary/Key Terms thực sự có đáp án.
- [ ] Thử context top-k 3 và 4 với cùng budget. Chỉ thử top-k 6 sau khi thấy thiếu evidence; không mặc định thêm context sẽ tăng precision.
- [ ] Kiểm tra PDF page và printed page ở nhiều chương, đầu/cuối sách và điểm chuyển mục. Giữ offset -12 nếu được xác nhận; không đổi offset chỉ để khớp một ví dụ trong mô tả.
- [ ] Sửa metadata section theo heading ở cấp đoạn nếu TOC gần đúng gán sai; tạo artifact mới và lưu bằng chứng mapping.
- [ ] Xác minh hợp đồng tên section. CSV hiện dùng `Chapter .../1.2 ...`, ví dụ cuộc thi dùng slug như `psychological_research/approaches_to_research`. **Chưa có bằng chứng bắt buộc slug hay exact-match**, nên không đổi hàng loạt bằng lower-case/underscore.
- [ ] Chỉ dùng mapping canonical được xác minh từ dữ liệu/scorer chính thức hoặc bảng annotation; ghi `section_id` riêng với tên hiển thị.
- [ ] References lấy từ metadata của evidence thực sự hỗ trợ answer, không do LLM bịa số trang. Đảm bảo mỗi mệnh đề cần nguồn có ít nhất một reference, đồng thời đo reference dư.

Nếu thay context cuối cùng bằng evidence đã rút gọn, phải chạy lại generation trên chính context đó và audit lần cuối. Không xuất một context khác với context thật đã tạo ra answer. Nếu chỉ rút gọn references, giữ trace giải thích vì sao một chunk trong prompt không được trích dẫn.

Giữ provenance PDF/section/page/span xuyên suốt retrieval, packing, generation và export. CSV vẫn là `ID` số nguyên, `context`/`answer` chuỗi, `references` JSON với `sections` là danh sách chuỗi và `pages` là danh sách số nguyên theo Dataset Description người dùng cung cấp. Kiểm tra số học dạng số trong CSV bằng parser, không dựa vào việc CSV có kiểu dữ liệu nội tại.

## 7. P3 — Tạo bộ SFT phục vụ trả lời có nguồn

### Quy mô và cấu trúc

Pilot 200–300 mẫu train đã review để kiểm chứng pipeline. Sau khi pilot đạt gate, mở rộng khoảng **800–1.200 mẫu train**, giữ nguyên dev/holdout độc lập. Các con số là ngân sách ban đầu có thể điều chỉnh, không phải ngưỡng bảo đảm tăng điểm.

Phân bổ khởi đầu: 25% định nghĩa/chức năng; 25% so sánh; 20% cơ chế/quy trình; 20% liệt kê/tổng hợp nhiều đoạn; 10% evidence thiếu hoặc bị nhiễu. Điều chỉnh theo bộ câu hỏi phát triển từ sách, không theo đáp án private.

### Quy trình tạo nhãn

1. Chọn source fact và source spans trước; ghi đúng quan hệ, năm, tên và giới hạn của khẳng định.
2. Sinh câu hỏi từ các fact đó bằng model local công khai ≤3B. Câu hỏi phải tự đứng được và không cần nhìn đoạn nguồn mới hiểu.
3. Reviewer xác nhận câu hỏi trả lời được; tạo đáp án ngắn theo các ý bắt buộc. Cho phép paraphrase và tổng hợp, không bắt toàn bộ answer là substring.
4. Dùng đúng retriever và `pack_context` production để tạo context train. Kết hợp trường hợp nguồn sạch với 2–4 chunk có đoạn nhiễu liên quan; không chỉ một chunk oracle như trước.
5. Đảm bảo nhãn answer có nguồn trong context thực tế. Khi truy hồi thiếu, sửa retrieval hoặc tạo nhãn chỉ trả lời phần được hỗ trợ; không giữ nhãn đầy đủ dựa trên đoạn bị giấu khỏi prompt.
6. Lưu ánh xạ mỗi mệnh đề trong answer tới evidence span. Kiểm tra độc lập câu hỏi–đáp án, coverage và mâu thuẫn; không dùng substring/semantic similarity làm bộ xác nhận duy nhất.
7. Tách group và loại trùng trước khi tạo paraphrase; lưu các cặp bị loại với lý do. Không dùng dev/holdout để huấn luyện verifier.

Ngưỡng bắt đầu train pilot đề xuất: đủ 200 mẫu train và 80 dev đã review; không trùng group giữa các split. **100% mẫu đưa vào train phải có trạng thái approved và source mapping.** Audit độc lập một mẫu ngẫu nhiên, có phân tầng theo loại câu hỏi, cần đạt ít nhất 95% về độ đúng/đủ và không có lỗi sai nguồn nghiêm trọng. Nếu không đạt, sửa lô dữ liệu và audit lại trước train. Mốc 95% là gate audit chất lượng nhãn đã review, không có nghĩa cho phép 5% dữ liệu chưa review vào train.

Giữ một nhóm mẫu chống lỗi: đổi nhầm năm, ghép nhầm người/lý thuyết, chỉ có một vế của phép so sánh, context bài tập không có đáp án, đoạn nguồn mâu thuẫn. Gold phải chỉ rõ xử lý đúng theo sách. Không chuyển các lỗi quan sát từ 50 câu test thành nhãn train cho chính các câu đó.

Không yêu cầu generator xuất `references` trong SFT trả lời thông thường. Citation mapping là trường audit/phụ trợ. Nếu thử cho model chọn evidence ID, triển khai parser riêng, kiểm tra ID nằm trong context rồi resolve metadata; không cho model viết số trang tự do.

## 8. P4 — Fine-tune có theo dõi đúng hành vi

Giữ Qwen2.5-1.5B-Instruct và revision base đã ghim. Loss assistant-only, giữ end-of-turn và mask prompt/padding như hiện tại. SFT vẫn tối ưu token cross-entropy; “theo metric” ở đây nghĩa là **thiết kế nhãn, đánh giá và chọn checkpoint theo bốn thành phần**, chưa phải tối ưu trực tiếp scorer bí mật bằng gradient.

| Hạng mục | Cấu hình thử đầu đề xuất |
|---|---|
| LoRA modules | `q_proj,k_proj,v_proj,o_proj` |
| Rank/alpha | r=8, alpha=16; chỉ thử r=16 sau khi dữ liệu/evaluation ổn |
| Learning rate | Pilot đầu 1e-5; sau đó thử 3e-5 trong run riêng nếu cần |
| Dropout | 0,05 |
| Epoch | 1 trước; thử 2 chỉ khi dev cải thiện và không có suy giảm faithfulness |
| Effective batch | 8; ghi rõ `per_device_batch × accumulation × world_size` |
| Precision | BF16 nếu hỗ trợ; nếu không thì FP16 như pipeline hiện tại |
| Sequence length | 2.048 ban đầu; đo phân phối token, tăng trong giới hạn model nếu cần, không cắt mất gold answer |
| Evaluation | Sinh answer thực trên cùng dev tại step 0, các checkpoint và cuối train; đo CP/AF/AC/RA |
| Model selection | Đánh giá checkpoint trên dev đã khóa, giữ ứng viên đạt gate và có `S_proxy` tốt nhất; loss chỉ là số chẩn đoán |

### Cấu hình pilot và giới hạn triển khai

Pilot đầu: 200–300 train approved, 80 dev, một GPU, batch/device=1, accumulation=8, một epoch, LoRA r=8/alpha=16, learning rate 1e-5, dropout 0,05 và seed 42. Chạy baseline dev trước khi cập nhật weight. Dùng cùng system/user template, tokenizer và cách `pack_context` cho train và inference.

Cấu hình trên là điểm khởi đầu để kiểm chứng, không phải siêu tham số đã chứng minh tối ưu. Không đổi rank, learning rate, prompt, context top-k và số epoch cùng lúc. Nếu pilot không qua gate, xem lỗi theo loại câu hỏi trước khi thử run mới; nếu dữ liệu có vấn đề thì sửa dữ liệu trước.

Trainer cũ chọn bằng `metric_for_best_model="eval_loss"`. Code v2 đã thay bằng vòng train kiểm đếm mẫu/update và sinh đáp án dev ở step0, giữa epoch và cuối epoch. Train xong vẫn phải chờ reviewer chấm dev trước khi chọn checkpoint, xác nhận holdout rồi merge; không chọn bằng loss thay cho proxy.

Với pilot ngắn, đánh giá step 0 và cuối epoch là tối thiểu. Với bộ lớn, thêm checkpoint giữa epoch; chọn lịch dựa trên số optimizer steps thực, tránh đặt `eval_steps` lớn hơn toàn bộ lần train. Số checkpoint giữ lại phải đủ cho lịch đánh giá dự kiến, không xóa checkpoint trước khi chấm xong.

Ưu tiên một GPU cho pilot để dễ kiểm toán; không mặc định DataParallel hai GPU tăng độ tin cậy. Trước train, in `num_examples`, số batch/epoch, optimizer steps dự kiến, effective batch, warmup steps, số GPU. Sau train, lưu `trainer_state.json`, `global_step`, số mẫu đã xử lý và config thực tế.

Điều tra riêng vì sao lần hiện tại báo `2/2` và epoch 0,762: kiểm tra Trainer 4.51.3, batch thực tế, xử lý nhóm accumulation cuối, dataloader và thiết bị. Không khẳng định đã chạy đủ một epoch chỉ dựa vào `num_train_epochs: 1`. Không tăng `max_steps` mù quáng để che lỗi đếm mẫu.

Với 800 mẫu và effective batch 8, khoảng 100 optimizer steps/epoch là kỳ vọng để đối chiếu; số thực tế phụ thuộc batching và phần dư. Không coi đạt 100 steps là bằng chứng model tốt. Log learning rate theo step để thấy lịch warmup/cosine có hoạt động trên số step thực không.

Lưu base, adapter và model merged. Kiểm tra trên một tập prompt cố định rằng adapter chưa merge và model merged cho output tương đương trong dung sai số học/decoding đã định; verify checksum và model identity trong generation diagnosis.

Không bắt đầu DPO/RL chỉ để “tối ưu metric” khi chưa có gold/rubric ổn định. Nếu SFT chất lượng cao vẫn có lỗi lặp lại, cân nhắc preference pairs được review trong pha sau, với chosen/rejected cùng question/context và không lấy test làm training signal.

## 9. P5 — Ma trận thí nghiệm và điều kiện thay baseline

| Run | Generator | Retrieval/context/references | Mục đích |
|---|---|---|---|
| E0 | Qwen gốc | Policy hiện tại, khóa input | Baseline có thể tái lập |
| E1 | LoRA hiện tại, artifact `590f60...` | Giống hệt E0 | Đo riêng tác động lần fine-tune vừa chạy |
| E2 | SFT mới, dữ liệu đã review | Giống hệt E0 | Đo riêng tác động SFT mới, chưa trộn với sửa retrieval |
| E3 | Qwen gốc | Evidence/metadata đã sửa từ P2 | So với E0 để đo tác động retrieval/reference |
| E4 | Cùng model SFT của E2 | Giống hệt E3 | So với E3 để đo lợi ích adapter trong pipeline mới; so với E2 để xem tác động đổi retrieval |
| E5 | Như E2, chỉ đổi learning rate hoặc một biến đã chọn | Giống hệt E0 | Kiểm tra độ nhạy nếu phân tích pilot cho thấy có lý do |

E0/E1/E2 dùng cùng frozen prompt/context/references theo từng query. E3/E4 dùng bộ input đã sửa thứ hai, giống nhau trong cặp. Không so E4 với E0 rồi quy toàn bộ chênh lệch cho fine-tune. Nếu retrain SFT với context policy mới sau P2, tạo run/model ID riêng và so lại với E3, không ghi đè model của E2.

Mỗi run lưu per-query metrics, phân phối theo loại câu hỏi, prompt/context hash, model/config/data hash, latency và số lỗi. Chấm answer ẩn tên model. Dùng paired bootstrap theo question group để báo khoảng tin cậy cho chênh lệch proxy; dev nhỏ có thể chưa đủ kết luận. Xác nhận ứng viên bằng seed thứ hai và holdout một lần; không so loss giữa các tập khác nhau.

**Gate đề xuất, không phải ngưỡng của Kaggle:**

- [ ] Schema đạt 100%, không mất ID, không answer/context rỗng, references parse được.
- [ ] Không có lỗi nghiêm trọng mới về năm, quan hệ hoặc attribution trong tập audit đã khóa.
- [ ] AC và AF trên dev không giảm; báo cả mức tăng và độ bất định.
- [ ] CP và RA không giảm khi thay generator; nếu thay pipeline thì đánh giá lại độc lập.
- [ ] `S_proxy` tăng ít nhất 0,02 trên thang 0–1 và cận dưới khoảng tin cậy chênh lệch vượt 0 để quyết định tự động. Nếu chưa đủ mẫu, kết luận “chưa đủ bằng chứng” và giữ baseline.
- [ ] Holdout xác nhận chiều cải thiện, không xuất hiện suy giảm lớn ở loại câu hỏi quan trọng.
- [ ] Kiểm tra model đã merge và inference offline thành công.

Áp dụng gate với **baseline có cùng policy**: E2 so E0; E4 so E3. Để chọn pipeline cuối cùng, báo thêm so sánh end-to-end với E0 và giữ bản tốt nhất đã xác nhận. Mức tăng 0,02 là ngưỡng đề xuất cần chốt trước thí nghiệm; không hạ ngưỡng sau khi xem kết quả để hợp thức hóa một model.

Khoảng tin cậy đề xuất là paired bootstrap 95% theo group, cùng số lần resample và seed đã cố định. Đây là ước lượng trên tập gán nhãn, không phải bảo đảm cho private score. Dev dùng chọn checkpoint nên có thiên lệch chọn; holdout mới là bước xác nhận độc lập, còn seed thứ hai kiểm tra độ ổn định của huấn luyện.

Nếu model fine-tune không vượt gate, tiếp tục dùng baseline và giữ artifact để nghiên cứu. Không tự chọn model mới nhất hoặc model có `eval_loss` thấp nhất. Có thể fine-tune làm điểm thấp hơn; không cam kết phục hồi hay vượt một mức private cụ thể trước khi đo.

## 10. P6 — Đóng gói và kiểm tra submission

- [ ] Tách notebook chuẩn bị dữ liệu/train khỏi notebook submission inference; gắn model, tokenizer, code, dependency và dữ liệu cần thiết làm Kaggle Input.
- [ ] Submission inference chạy Internet OFF, không `git clone`, `pip install` qua mạng hay gọi API bên ngoài. Nếu cần cài package thì dùng wheel offline đã kiểm tra.
- [ ] Chạy trong giới hạn 9 giờ của điều kiện notebook được nêu trong Overview; đo cả model load, retrieval và export.
- [ ] Tất cả model dùng trong giải pháp tuân thủ giới hạn công bố ≤3B và yêu cầu công khai/miễn phí; không thay teacher lớn hơn mà bỏ qua ràng buộc này.
- [ ] Giữ tên `submission.csv`; 50 ID của bộ hiện tại phải khớp queries, không thiếu/trùng; xác minh lại số câu khi input đổi.
- [ ] `context` phải là text sách đã dùng; `answer` ngắn nhưng đủ ý; `references` JSON đúng section/page có nguồn.
- [ ] Lưu report phân biệt `schema_verified`, `proxy_evaluated`, `official_metric_verified` và `kaggle_score_reported`; không dùng một cờ `valid` cho mọi ý nghĩa.

Theo nguồn người dùng cung cấp, metadata dataset là CC BY-NC-SA 4.0, còn phần mô tả sách ghi OpenStax Psychology 2e CC BY 4.0. Ghi riêng hai nguồn/giấy phép trong provenance thay vì gộp thành một nhãn cho tất cả dữ liệu và artifact.

## 11. Các đầu việc triển khai và điều kiện hoàn tất

| Ưu tiên | File/module dự kiến | Kết quả cần có để hoàn tất |
|---|---|---|
| P0 | Script audit A/B mới; notebook selector | Có baseline tái lập và bảng diff; không tự chuyển sang model fine-tune |
| P1 | `casml_b0/evaluation.py` mới, `configs/eval.yaml`, dữ liệu dev/holdout | Spec proxy, rubric, annotation, per-query scoring, bootstrap; ghi rõ không phải metric chính thức |
| P2 | `retrieval.py`, `context.py`, `prepare.py`, `exporting.py` | Coverage nhiều vế, bớt evidence nhiễu, mapping nguồn được kiểm tra |
| P3 | `synthetic.py` hoặc builder SFT mới | Source facts + review + retrieved context; group split; kiểm tra nhãn bằng mệnh đề |
| P4 | `finetuning.py`, config, notebook | Cấu hình pilot v2, log optimizer steps, đánh giá generated answers, checkpoint selection theo gate, kiểm tra merge |
| P5 | Experiment runner/report | E0–E4 giữ biến kiểm soát; bảng metric và quyết định chọn model có bằng chứng; E5 chỉ chạy khi cần |
| P6 | Notebook inference offline | Xuất đủ CSV đúng hợp đồng trong runtime cho phép |

Kiểm thử cần bổ sung tập trung vào rủi ro thực: rò rỉ group giữa train/holdout; loss mask; số optimizer step và phần dư; context có đủ hai vế; nhầm quan hệ dù từ khóa tồn tại; citation mapping; input A/B giống nhau; artifact merge; chạy offline. Unit test đạt chỉ xác minh cơ chế, **không phải bằng chứng điểm private sẽ tăng**.

Thứ tự thực hiện: **P0 → P1 → P3 → pilot P4/E2 → P5 → P6**. P2/E3 là nhánh sửa retrieval trên baseline; ghép với generator mới ở E4 khi cả hai nhánh đã có báo cáo riêng. Chưa mở rộng sinh thêm hàng nghìn cặp theo policy sao chép hiện tại.

### Lô triển khai đầu tiên

| Bước | Đầu vào cần có | Đầu ra cần giao | Điều kiện chuyển bước |
|---|---|---|---|
| 1. Tách train và lựa chọn model | Repo hiện tại, model base, manifest run 0.330 | Notebook có model selector rõ ràng; baseline không kích hoạt synthetic/train | Chạy lại section inference không làm phát sinh train hoặc đổi model |
| 2. Khóa dữ liệu và evaluator | Corpus sách, rubric ở mục 5 | Dataset draft + bảng review + split manifest; evaluator spec có version | Có gold dev đã kiểm tra; không rò rỉ group; evaluator không dùng test answer |
| 3. So sánh E0/E1 | Base + adapter hiện tại, input đã khóa | Báo cáo A/B với hash và per-query lỗi/metric | Xác nhận hai run chỉ khác model; phần chưa có gold ghi chưa đánh giá |
| 4. SFT pilot E2 | Train approved và dev độc lập | Adapter/checkpoints, training report, generated dev answers | Số bước thực khớp lịch dự kiến; nhãn và prompt được audit |
| 5. Quyết định model | Báo cáo dev/holdout, kiểm tra merge | Báo cáo chọn hoặc từ chối ứng viên kèm lý do | Qua gate mới chọn model; không qua thì baseline tiếp tục được dùng |

Đường dẫn artifact dự kiến, chưa phải file đã tạo:

```text
artifacts/sft_reviewed_v1/{drafts,annotations,train,dev,holdout}.jsonl
artifacts/sft_reviewed_v1/split_manifest.json
eval/metric_spec_v1.md
eval/e0_base_locked/
eval/e1_current_lora_locked/
eval/e2_reviewed_sft_locked/
artifacts/qwen15b_reviewed_lora_v1/
eval/model_selection_v1.json
```

Gold holdout phải được tách quyền đọc trong quy trình đánh giá: trainer và checkpoint selector chỉ nhận train/dev; script đánh giá cuối mới nhận holdout. Có file trong cùng máy không đồng nghĩa được phép đưa nó vào callback đánh giá ở mọi epoch.

Nội dung tối thiểu của `model_selection_v1.json`: model/artifact ID của ứng viên và baseline, dataset/evaluator version, bảng bốn proxy, chênh lệch và khoảng tin cậy, kết quả các gate, trạng thái `selected/rejected/insufficient_evidence`, cùng lý do. Đây là schema đề xuất cho bước triển khai, chưa được hỗ trợ sẵn trong CLI hiện tại.

## 12. Nguồn và giới hạn kiểm chứng

- [Log lần chạy](<C:/Users/HP/Downloads/casml-full-run (1).log>): dòng 1655–1657 và report tiếp theo về dữ liệu; 4448–4455 về hai bước train; 4560–4579 về loss; 4599–4602 về model đang dùng; 5175–5183 về references và giới hạn verification.
- [Diagnosis](C:/Users/HP/Downloads/diagnosis.json): run ID `af68dc1d11214a74a243859e222c790cc17c4a1d5eb05f5515452379eca79d582`; artifact fine-tune, cấu hình, 50 câu và evidence thật.
- [Submission](C:/Users/HP/Downloads/submission.csv): đã kiểm tra cấu trúc, ID, JSON references và đối chiếu answer/context với diagnosis. Không có ground truth để xác nhận toàn bộ câu trả lời.
- [Overview người dùng dán](<C:/Users/HP/.codex/attachments/261f728c-1fcd-4ca6-8d39-8f2af152ad46/Pasted text.txt>) và Dataset Description trong tin nhắn: nguồn của trọng số, ràng buộc, kiểu pages và schema.
- [CASML Overview](https://www.kaggle.com/competitions/casml-generative-ai-hackathon/overview), [CASML Data](https://www.kaggle.com/competitions/casml-generative-ai-hackathon/data): đã mở để đối chiếu nhưng trang động không trả nội dung chi tiết cho công cụ đọc web; không tuyên bố đã đọc mã metric hoặc ground truth từ hai trang này.
- Mã dự án đã đọc: `synthetic.py`, `finetuning.py`, `exporting.py` và các config liên quan. Chưa nhận được đầy đủ artifact SFT, `trainer_state.json`, source PDF của lần chạy hoặc CSV baseline để phân tích nguyên nhân nhân quả.

SHA256 của các file đầu vào tại thời điểm lập kế hoạch:

```text
diagnosis.json
0939cef238c2c3897c7cd78277b4ed5818c6f5b10217fba31d79d40f50562c0c
submission.csv
d48cce23977c2553e83da02deb047be6f301091278de35aec25fe14e71a8dbe4
casml-full-run (1).log
1715c141b13f16e4df2b96a2e455dc165c149358daabe80237e30aa1c58f8193
```
