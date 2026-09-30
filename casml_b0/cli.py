"""Thin command router with lazy imports, preserving stage independence."""
import argparse
from .artifacts import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="CASML B0 — independent artifact-based RAG stages")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "index", "retrieve", "generate", "export"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--out", required=True)
        if name == "prepare":
            sub.add_argument("--pdf", required=True)
            sub.add_argument("--page-map-override")
        elif name == "index":
            sub.add_argument("--corpus", required=True)
        elif name == "retrieve":
            sub.add_argument("--index", required=True)
            sub.add_argument("--queries", required=True)
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
    elif args.command == "generate":
        from .generation import generate
        m = generate(args.retrieval, config, args.config, args.out, args.limit)
    else:
        from .exporting import export
        m = export(args.run, args.queries, config, args.out, args.sample, args.pdf)
    print(f"{m['stage']}: {args.out} | {m['artifact_id'][:12]} | complete={m['complete']}")
