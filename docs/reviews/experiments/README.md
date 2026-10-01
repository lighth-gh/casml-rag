# Draft ablations from Version 11

These drafts do not change production defaults. No generation or Kaggle submission has been run.

Run from the repository root and reuse the actual full hybrid retrieval artifact (30 candidates/query). The local examples/verified artifact is a demo, not the competition corpus. Use fresh run/output directories.

Example in Kaggle after cloning the repository containing these drafts:

    python -m casml_b0 generate --retrieval /kaggle/working/casml_b0_work/artifacts/retrieval_hybrid_r2 --config docs/reviews/experiments/e0_v11_baseline.yaml --out /kaggle/working/casml_b0_work/runs/review_e0_v11_baseline
    python -m casml_b0 export --run /kaggle/working/casml_b0_work/runs/review_e0_v11_baseline --queries "/kaggle/input/competitions/casml-generative-ai-hackathon/Dataset_RAG (1)/queries.json" --config docs/reviews/experiments/export_report_only.yaml --pdf "/kaggle/input/competitions/casml-generative-ai-hackathon/Dataset_RAG (1)/book.pdf" --out /kaggle/working/casml_b0_work/outputs/review_e0_v11_baseline

For E1/E2/E3 replace the generation config and both run/output names. E1 changes only the system prompt. E2 changes only context_top_k 4→3. E3 changes only context_top_k 4→6. Budget stays 1900. Compare all 50 rows, actual evidence and manual checklist.

Generation disables grounding_validator_enabled. Export keeps fail_on_truncation=true and sets fail_on_unsupported_claims=false. Numeric/name/word-count issues remain reported in validation.json without preventing export. This is experimental report-only handling, not factual certification.

The updated notebook now supports baseline/attribution/top3/top6 via EXPERIMENT and keeps grounding report-only. Prefer the maintained configs in configs/experiments for new runs; the YAML files in this review folder are the earlier standalone drafts.

E0 preserves Version 11 effective model/revision, prompt, initial decoding, EOS and token budget. Matching submission11 is a check to perform on the real generation run; config loading alone does not prove reproducible answers.
