"""Self-contained generation diagnostics, also available for incomplete runs."""
from datetime import datetime, timezone

from .artifacts import SCHEMA, write_json


def write_diagnosis(out, manifest, rows, results, *, state, runtime, resumed=0,
                    model_context_window=None, full_query_count=None, failure=None):
    queries = []
    for row in rows:
        result = results.get(row["query_id"])
        queries.append({k: v for k, v in result.items() if k != "record_sha256"} if result else
                       {"query_id": row["query_id"], "question": row["question"], "status": "pending"})
    issues = {
        "errors": [q["query_id"] for q in queries if q["status"] == "error"],
        "length_limited": [q["query_id"] for q in queries if q.get("finish_reason") == "length"],
        "insufficient_context": [q["query_id"] for q in queries if q["status"] == "insufficient_context"],
        "retried_for_length": [q["query_id"] for q in queries if len(q.get("generation_attempts", [])) > 1],
        "unfinished": [q["query_id"] for q in queries if q["status"] in ("pending", "running")],
    }
    write_json(out / "diagnosis.json", {
        "schema": SCHEMA, "kind": "generation_diagnosis", "run_id": manifest["artifact_id"],
        "state": state, "created_utc": manifest["created_utc"],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "signature": manifest["signature"], "environment": runtime,
        "model_context_window": model_context_window, "failure": failure,
        "summary": {"query_count": len(rows), "full_retrieval_query_count": full_query_count,
                    "resumed": resumed, **issues,
                    "generation_ready_for_export": state == "completed" and len(rows) == full_query_count
                    and not issues["errors"] and not issues["length_limited"] and not issues["unfinished"]},
        "queries": queries,
        "note": "Contains actual prompts, evidence, answers and completed attempts. Generation readiness "
                "does not validate factual accuracy or submission format. If the process is killed, "
                "state may remain running; this is the last saved snapshot.",
    })
