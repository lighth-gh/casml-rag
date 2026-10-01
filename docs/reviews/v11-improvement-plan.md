# Review chất lượng Version 11 và hướng cải tiến

Nguồn: diagnosis11.json, submission11.csv, casml-r1 (4).log người dùng xác nhận thuộc Version 11. Tất cả 50 answers đã đọc; các findings dưới đây được đối chiếu với selected evidence. Checklist là diagnostic development set, không phải official gold/metric; chưa chấm accuracy toàn bộ 50 câu.

## Kết luận ưu tiên

Giữ Version 11 làm baseline 0.317 public / 0.334 private. Hướng cải tiến có căn cứ nhất là giữ đúng entity/topic attribution trong generation, chọn context theo coverage và chức năng nội dung, sửa extraction/chunk boundaries khi xem được PDF gốc, rồi thử model lớn hơn trên evidence cố định. Mỗi thay đổi chạy riêng; chưa có căn cứ bảo đảm tăng leaderboard.

## Chỉnh lại nhận định Q12

Version 11 gán functionalism cho James đúng. Version 13 xóa câu giới thiệu James có birth year sai rồi nối các câu còn lại: “He was a proponent of the functionalist perspective” nằm sau đoạn Wundt, mất antecedent James. Đây là regression do xóa câu. Dream analysis/slips of tongue/free association gán cho James đã sai từ Version 11. Báo cáo r13-regression-review.md đã được chỉnh lại.

## Các lỗi cụ thể

| Nhóm | Bằng chứng Version 11 | Hướng sửa |
|---|---|---|
| Gán nhầm người/chủ đề | Q12 lấy phương pháp Freud gán James; Q17 gán đoạn humanism cho Skinner; Q19 gán biomedical-treatment paragraph cho client-centered therapy; Q26 gán postformal thought cho Piaget; Q45 lấy kết luận chọn trường của Prosser gán Clarks | Prompt giữ fact gắn đúng người/theory/therapy; giữ heading boundaries; chấm entity–attribute |
| Sai số/tên dù nguồn có sẵn | Q11 1874 thay 1873; Q12 1889 thay 1879 và 1948; Q31 568/22013, trộn diagnosis count vào year; Q30 American Psychologist thay American Psychiatric Association | Chỉ dùng số khi cần và đúng quan hệ; copy canonical terms; numeric checks báo lỗi thay vì xóa câu |
| Sai thuật ngữ/ngôn ngữ | Q8 có “và”, conscienctiousness/extrovertion; Q35 cross-sectional thay cross-cultural | English-only/canonical-spelling checks; prompt thuật ngữ; model ablation |
| Optional-detail hallucination | Q16 thêm electric shock trong ví dụ Little Albert ngoài context; OpenStax chính thức mô tả loud sound | Trả lời đủ định nghĩa từ evidence; giảm ví dụ không có nguồn |
| Thiếu coverage | Q48 cả 4 chunks về personality disorders, thiếu mood-disorder definition; Q26 thiếu chi tiết nhiều stages | Split query theo các phần cần trả lời; bảo đảm evidence mỗi vế; neighbor expansion |
| Nhiễu context | Q22 rank 1 là bibliography p638; Q31 có bibliography p716; Q33 có bibliography p634; Q50 có exercise questions | Gắn content role; giảm ưu tiên references/exercises theo query; giữ định nghĩa hữu ích trong body/summary/glossary |
| Chunk trộn section | Q17 chunk chuyển Skinner → Maslow/Rogers/Humanism; Q19 summary chuyển Rogerian → biological treatments | Section/paragraph-aware chunks; metadata đúng span thay vì TOC label theo page |
| Extraction mất khoảng trắng | PrinciplesofPhysiologicalPsychology, Brownv.BoardofEducation, JournalofPersonalityandSocialPsychology có trong nguồn | Xem PDF text/words/blocks; sửa join spans theo layout, tránh tách CamelCase mù |

Q16 được đối chiếu với [OpenStax 6.2](https://openstax.org/books/psychology-2e/pages/6-2-classical-conditioning). Nguồn web chỉ phục vụ review, không đưa vào pipeline/corpus/prompt cuộc thi.

## Vấn đề ở context packing

50/50 câu chọn đúng 4 chunks, trong khi cache có 30 candidates. Context_tokens min 428, max 1331, trung bình 1077.2, dưới budget 1900. context_top_k=4 dừng lựa chọn trước khi dùng hết budget. Bổ sung evidence có thể giúp coverage; thêm nhiều đoạn nhiễu có thể làm quality giảm.

Diagnosis chỉ chứa 4 selected chunks, chưa thấy 26 candidates còn lại. Cần xem retrieval.jsonl để phân biệt relevant evidence thiếu khỏi top30 hay chỉ rơi ngoài top4. Không cần rebuild pipeline cho ablation prompt/top-k.

Thứ tự thử context:

1. So top3/top4/top6 trên cùng cache, model và prompt. Top3 kiểm tra noise; top6 kiểm tra coverage. So actual selected evidence, không chỉ số từ.
2. Selector theo coverage cho comparison/multi-entity: retrieve mỗi vế, union/dedup, rerank và giữ nguồn hỗ trợ mỗi vế. Q48 cần mood + personality. Không hardcode IDs hoặc gold answers.
3. Adjacent paragraph/page expansion trong same section khi hit bắt đầu giữa câu hoặc thiếu định nghĩa/stage list. Giữ source/page từng span và dedup overlap.
4. Content-role metadata cho reference/exercise material. Summary và glossary có thể chứa định nghĩa tốt nên đánh giá riêng.
5. Đánh giá relevance trước khi dùng confidence threshold: Q45 source p304 chứa Clark doll-choice evidence phù hợp nhưng reranker score khoảng −6.56. Ngưỡng score>0 sẽ loại nguồn này.

## Ablation đề xuất

| Thứ tự | Thí nghiệm | Yếu tố đổi | Giữ cố định | Tiêu chí |
|---|---|---|---|---|
| E0 | Policy bản 11 trên current code | Grounding output mutation off; exporter report-only | Model/revision/prompt/top4/decoding | Đối chiếu submission11, ghi mọi khác biệt |
| E1 | Source-attribution prompt | Chỉ system prompt | Retrieval/top4/model/decoding | Q12/17/19/26/35/45 bớt nhầm; controls đủ ý |
| E2 | Context top3 | Chỉ top_k 4→3 | Prompt baseline/budget1900/model | Noise giảm mà key facts không mất |
| E3 | Context top6 | Chỉ top_k 4→6 | Prompt baseline/budget1900/model | Coverage Q26/Q48 cải thiện, không tăng hallucination |
| E4 | Coverage-aware packing | Chỉ selection policy | Model/prompt/candidate cache | Mỗi vế có support; source pages chính xác |
| E5 | Model ablation | Generator model + revision của model đó | Evidence/prompt/decoding tương đương | Ít numeric/entity/term errors hơn; VRAM/runtime phù hợp |

E1/E2/E3 chạy riêng từ E0; sau khi chọn thay đổi thắng mới thử kết hợp và kiểm tra interaction.

Model candidate: [Qwen2.5-3B-Instruct chính thức](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), model card mô tả 3.09B parameters và load qua Transformers. Đây là ứng viên thử, chưa có benchmark CASML chứng minh hơn 1.5B. Pin đúng revision của 3B, kiểm tra fit GPU và actual evidence; không tái dùng revision hash của 1.5B.

Chỉ thử embedding/reranker/HyDE sau khi kiểm tra top30 và thấy candidate recall thiếu. Các lỗi có source đúng sẵn cần được xử lý ở generation trước.

## Kiểm chứng chất lượng

Checklist 14 diagnostic queries trong v11-review-checklist.json gồm required concepts, forbidden confusions và source pages. Chấm thủ công:

- Core correctness: đạt/không đạt.
- Supported coverage: số ý cần trả lời có nguồn và đã được trả lời / số ý cần trả lời.
- Unsupported/misattributed claims: đếm theo claim; token xuất hiện đâu đó trong context không đủ làm support.
- Output health: language, spelling, duplicate ideas, cutoff, dangling pronouns.

Controls đề xuất: 1,4,9,15,24,25,34,46; cần đọc evidence trước khi coi là clean. Kiểm tra cả 50 câu sau mỗi candidate. Thêm held-out textbook queries với source anchors/required concepts do người đọc đánh dấu; không dùng model answer làm gold hoặc dùng 14 câu đã tune làm holdout.

7/50 answers vượt 140 từ: Q12,14,36,42,45,48,50. Thử prompt concision riêng và đo completeness; không cắt prefix để ép length pass. Numbers/names validator chỉ dùng triage, không dùng pass rate thay accuracy.

## Deliverables và giới hạn

Bốn YAML E0–E3, exporter report-only và prompt ứng viên ở docs/reviews/experiments. Đã load configs, xác minh paths, prompt baseline, model revision, token settings; chưa chạy LLM/generation và chưa có score mới.

Notebook hiện tại hardcode grounding_validator_enabled=True và export gate True. Drafts dùng CLI độc lập; triển khai notebook cần experiment mode thay overrides tương ứng. Xem experiments/README.md.

Review chỉ tạo/cập nhật docs/reviews, không sửa production code/config/notebook hoặc push.
