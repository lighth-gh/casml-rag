"""Thin command router with lazy imports, preserving stage independence."""
import argparse
from .artifacts import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="CASML B0 — independent artifact-based RAG stages")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "index", "retrieve", "build-sft", "attach-review-context", "prepare-sft",
                 "finetune", "evaluate", "score-eval", "select-model", "merge-model", "generate", "export"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--out", required=True)
        if name == "prepare":
            sub.add_argument("--pdf", required=True)
            sub.add_argument("--page-map-override")
        elif name in ("index", "build-sft"):
            sub.add_argument("--corpus", required=True)
        elif name == "retrieve":
            sub.add_argument("--index", required=True)
            sub.add_argument("--queries", required=True)
        elif name == "finetune":
            data = sub.add_mutually_exclusive_group(required=True)
            data.add_argument("--dataset", help="Complete reviewed_sft artifact from prepare-sft")
            data.add_argument("--train", help="Legacy alias: now requires a reviewed_sft DIRECTORY")
            sub.add_argument("--validation", help="Optional separate validation JSONL")
        elif name == "prepare-sft":
            sub.add_argument("--corpus", required=True)
            sub.add_argument("--annotations", required=True)
        elif name == "attach-review-context":
            sub.add_argument("--drafts", required=True)
            sub.add_argument("--retrieval", required=True)
        elif name == "evaluate":
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--split", choices=("dev", "holdout"), required=True)
            sub.add_argument("--context-mode", choices=("retrieved", "oracle"), default="retrieved")
            sub.add_argument("--training")
            sub.add_argument("--checkpoint")
        elif name == "score-eval":
            sub.add_argument("--predictions", required=True)
            sub.add_argument("--reviews", required=True)
        elif name == "select-model":
            sub.add_argument("--baseline", required=True)
            sub.add_argument("--candidate", required=True)
            sub.add_argument("--holdout-baseline")
            sub.add_argument("--holdout-candidate")
        elif name == "merge-model":
            sub.add_argument("--training", required=True)
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--selection", required=True)
        elif name == "generate":
            sub.add_argument("--retrieval", required=True)
            sub.add_argument("--limit", type=int)
        elif name == "export":
            sub.add_argument("--run", required=True)
            sub.add_argument("--queries", required=True)
            sub.add_argument("--sample")
            sub.add_argument("--pdf", help="Optional original PDF; verifies hash and copies it beside audit.html")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        from .prepare import prepare
        m = prepare(args.pdf, config, args.out, args.page_map_override)
    elif args.command == "index":
        from .indexing import build_index
        m = build_index(args.corpus, config, args.out)
    elif args.command == "retrieve":
        from .retrieval import retrieve
        m = retrieve(args.index, args.queries, config, args.out)
    elif args.command == "build-sft":
        from .synthetic import build_sft
        m = build_sft(args.corpus, config, args.config, args.out)
    elif args.command == "finetune":
        from .finetuning import finetune
        m = finetune(args.dataset or args.train, config, args.config, args.out, args.validation)
    elif args.command == "prepare-sft":
        from .reviewed import prepare_reviewed
        m = prepare_reviewed(args.corpus, args.annotations, config, args.out)
    elif args.command == "attach-review-context":
        from .reviewed import attach_context
        m = attach_context(args.drafts, args.retrieval, config, args.config, args.out)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        m = evaluate(args.dataset, args.split, config, args.config, args.out, args.training, args.checkpoint,
                     context_mode=args.context_mode)
    elif args.command == "score-eval":
        from .evaluation import score_evaluation
        m = score_evaluation(args.predictions, args.reviews, config, args.out)
    elif args.command == "select-model":
        from .evaluation import select_model
        m = select_model(args.baseline, args.candidate, config, args.out, args.holdout_baseline, args.holdout_candidate)
    elif args.command == "merge-model":
        from .finetuning import merge_selected
        m = merge_selected(args.training, args.checkpoint, args.selection, config, args.out)
    elif args.command == "generate":
        from .generation import generate
        m = generate(args.retrieval, config, args.config, args.out, args.limit)
    else:
        from .exporting import export
        m = export(args.run, args.queries, config, args.out, args.sample, args.pdf)
    print(f"{m['stage']}: {args.out} | {m['artifact_id'][:12]} | complete={m['complete']}")
