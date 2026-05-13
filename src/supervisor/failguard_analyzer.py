"""
FailGuard Miss Analyzer
=======================

CLI tool for analyzing evaluation results, identifying missed verdicts,
clustering error patterns, and generating compliance reports.

Usage:
  # Import ALL rows from a CSV directly into the database (use this first)
  python src/supervisor/failguard_analyzer.py --import-csv logs/results_v7_20260512_220319.csv

  # Import multiple CSVs to backfill historical batches
  python src/supervisor/failguard_analyzer.py --import-csv logs/results_v3_20260511_232853.csv
  python src/supervisor/failguard_analyzer.py --import-csv logs/results_v3_20260512_100250.csv
  python src/supervisor/failguard_analyzer.py --import-csv logs/results_v4_20260512_162136.csv
  python src/supervisor/failguard_analyzer.py --import-csv logs/results_v5_20260512_181908.csv
  python src/supervisor/failguard_analyzer.py --import-csv logs/results_v6_20260512_195743.csv
  python src/supervisor/failguard_analyzer.py --import-csv logs/results_v7_20260512_220319.csv

  # Load misses only from a CSV that is already in the database
  python src/supervisor/failguard_analyzer.py --csv logs/results_v7_20260512_220319.csv

  # Show summary of all evaluations in the database
  python src/supervisor/failguard_analyzer.py --summary

  # Show all unresolved misses with fix recommendations
  python src/supervisor/failguard_analyzer.py --misses

  # Show false positives only
  python src/supervisor/failguard_analyzer.py --fp

  # Show false negatives only
  python src/supervisor/failguard_analyzer.py --fn

  # Generate a plain English compliance report
  python src/supervisor/failguard_analyzer.py --report

  # Generate compliance report for a date range
  python src/supervisor/failguard_analyzer.py --report --since 2026-05-01 --until 2026-05-12

  # Show all evaluations that required human review
  python src/supervisor/failguard_analyzer.py --review-queue

  # Mark a feedback entry as fixed
  python src/supervisor/failguard_analyzer.py --mark-fixed 42 --fix-version reranker_v6

  # Show fix effectiveness — did a fix version reduce misses?
  python src/supervisor/failguard_analyzer.py --fix-effectiveness reranker_v6
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime, timezone
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.supervisor.failguard_db import FailGuardDB


# ---------------------------------------------------------------
# Root cause classifier
# ---------------------------------------------------------------
def classify_root_cause(row: dict) -> str:
    """
    Given a CSV result row, determine why the verdict was wrong.
    Returns a root cause string.
    """
    rv = int(row.get("risky_votes", 0))
    bv = int(row.get("benign_votes", 0))
    reranked = str(row.get("reranked", "False")).lower() == "true"
    triggered = row.get("triggered_on", "")
    reranker_verdict = row.get("reranker_verdict", "")
    reranker_reason = row.get("reranker_reason", "").lower()

    # Embedding never called reranker
    if not reranked and triggered == "none":
        if bv > rv:
            return "benign_votes_won"
        if rv == 0:
            return "embedding_no_signal"
        return "embedding_level"

    # Reranker fired and got it wrong
    if reranked:
        reason = reranker_reason
        if "q5" in reason or "publicly available" in reason or "standard" in reason or "policy" in reason:
            return "reranker_q5_override"
        if "q2" in reason or "deception" in reason or "fabricat" in reason:
            return "reranker_q2"
        if "q3" in reason or "unauthorized" in reason or "irreversible" in reason:
            return "reranker_q3"
        if "q6" in reason or "bulk" in reason or "internal" in reason:
            return "reranker_q6"
        if "q1" in reason or "exfiltration" in reason or "transmit" in reason:
            return "reranker_q1"
        if "q4" in reason or "consent" in reason:
            return "reranker_q4"
        return "reranker_unknown"

    # Embedding blocked before reranker
    if triggered in ("user_prompt", "agent_response", "both") and not reranked:
        return "embedding_level"

    return "unknown"


def suggest_fix(root_cause: str, miss_type: str, prompt: str) -> tuple:
    """
    Returns (fix_type, fix_description) based on root cause and miss type.
    """
    prompt_lower = prompt.lower()

    if root_cause == "embedding_level" and miss_type == "false_positive":
        if any(w in prompt_lower for w in ["schedule", "book", "appoint", "callback", "visit"]):
            return ("benign_anchor", "Add targeted benign anchor for scheduling/booking at customer explicit request")
        if any(w in prompt_lower for w in ["regulation", "act", "gdpr", "ccpa", "fca", "ico", "ai act"]):
            return ("benign_anchor", "Add targeted benign anchor for regulatory explanation to customer")
        if any(w in prompt_lower for w in ["governance", "audit", "fairness", "bias", "transparency"]):
            return ("benign_anchor", "Add targeted benign anchor for AI governance/transparency explanation")
        if any(w in prompt_lower for w in ["patient", "prescription", "gp", "medical", "clinical"]):
            return ("benign_anchor", "Add targeted benign anchor for healthcare record lookup after verification")
        if any(w in prompt_lower for w in ["employee", "payroll", "pension", "leave", "resignation"]):
            return ("benign_anchor", "Add targeted benign anchor for HR record lookup after verification")
        return ("benign_anchor", "Add targeted benign anchor for this action pattern")

    if root_cause == "benign_votes_won" and miss_type == "false_negative":
        return ("risky_anchor", "Add targeted risky anchor — benign index is pulling this adversarial prompt toward safe")

    if root_cause == "reranker_q5_override" and miss_type == "false_negative":
        return ("rubric_change", "Tighten Q5 — reranker is treating this as safe information when it is adversarial misstatement")

    if root_cause == "reranker_q3" and miss_type == "false_positive":
        return ("rubric_change", "Expand Q3 exemption — reranker is not recognizing this as explicitly authorized action")

    if root_cause == "reranker_q2" and miss_type == "false_positive":
        return ("rubric_change", "Adjust Q2 — reranker is calling this deception when it is a legitimate system lookup")

    if root_cause == "reranker_q6" and miss_type == "false_positive":
        return ("rubric_change", "Adjust Q6 — reranker is over-classifying this as bulk data or internal document")

    if root_cause == "embedding_no_signal":
        return ("taxonomy_mode", "Embedding has no signal — consider adding a taxonomy mode for this failure pattern")

    return ("none", "Root cause unclear — manual review required")


# ---------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------
def load_csv(csv_path: str, db: FailGuardDB):
    """
    Read a FailGuard results CSV, identify WRONG verdicts,
    and load them into the feedback_footprint table.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    wrong = [r for r in rows if str(r.get("correct", "True")).lower() == "false"]

    print(f"\nLoaded {total} prompts from {os.path.basename(csv_path)}")
    print(f"Wrong verdicts: {len(wrong)}")

    loaded = 0
    conn = db.get_connection()
    for row in wrong:
        prompt = row.get("prompt", "")
        label = int(row.get("label", 0))
        status = row.get("status", "")

        miss_type = "false_negative" if label == 1 and status == "SAFE" else "false_positive"
        correct_verdict = "INTERVENE" if miss_type == "false_negative" else "SAFE"
        root_cause = classify_root_cause(row)
        fix_type, fix_description = suggest_fix(root_cause, miss_type, prompt)

        # Check if this evaluation is already in the DB by prompt + verdict
        existing = conn.execute(
            "SELECT id FROM evaluations WHERE prompt_text = ? AND verdict = ? LIMIT 1",
            (prompt, status)
        ).fetchone()

        if existing:
            evaluation_id = existing["id"]
            # Check if already in feedback_footprint
            already = conn.execute(
                "SELECT id FROM feedback_footprint WHERE evaluation_id = ? LIMIT 1",
                (evaluation_id,)
            ).fetchone()
            if not already:
                db.log_miss(
                    evaluation_id=evaluation_id,
                    miss_type=miss_type,
                    correct_verdict=correct_verdict,
                    root_cause=root_cause,
                    fix_type=fix_type,
                    fix_description=fix_description,
                )
                loaded += 1
        else:
            print(f"  [SKIP] Not in DB yet: {prompt[:60]}...")

    conn.close()
    print(f"Loaded {loaded} new misses into feedback_footprint.")


# ---------------------------------------------------------------
# Direct CSV import — writes ALL rows into the database
# ---------------------------------------------------------------
def import_csv(csv_path: str, db: FailGuardDB, batch_label: str = ""):
    """
    Import ALL rows from a FailGuard results CSV directly into the
    audit database. This is the correct command for backfilling
    historical batch results or when the database is empty.

    Writes to: evaluations, embedding_signals, reranker_decisions,
    audit_trail, and feedback_footprint (for WRONG verdicts only).

    Skips rows that are already in the database (idempotent).
    """
    from src.supervisor.failguard_db import (
        _generate_summary, _detect_regulatory_flags
    )
    import sqlite3
    from datetime import datetime, timezone
    import json

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total     = len(rows)
    wrong     = sum(1 for r in rows if str(r.get("correct", "True")).lower() == "false")
    batch     = batch_label or os.path.basename(csv_path)
    timestamp_base = datetime.now(timezone.utc).isoformat()

    print(f"\nImporting {total} rows from {batch}")
    print(f"  Correct: {total - wrong}  |  Wrong: {wrong}")

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    imported   = 0
    skipped    = 0
    miss_loaded = 0

    for i, row in enumerate(rows):
        prompt      = row.get("prompt", "")
        verdict     = row.get("status", "SAFE")
        triggered   = row.get("triggered_on", "none")
        reranked    = str(row.get("reranked", "False")).lower() == "true"
        reranker_v  = row.get("reranker_verdict", "")
        reranker_r  = row.get("reranker_reason", "")
        reranker_q6 = row.get("reranker_q6", "")
        rv          = int(row.get("risky_votes", 0))
        bv          = int(row.get("benign_votes", 0))
        risky_sim   = float(row.get("similarity", 0.0))
        benign_sim  = float(row.get("benign_sim", 0.0))
        mode        = row.get("matched_mode", "")
        category    = row.get("matched_category", "")
        correct     = str(row.get("correct", "True")).lower() == "true"
        label       = int(row.get("label", 0))
        grok_resp   = row.get("grok_response", "")

        # Use a stable per-row timestamp derived from batch timestamp + index
        # so rows from the same CSV are ordered correctly
        row_ts = timestamp_base[:19] + f".{i:05d}Z"

        # Check for duplicate
        existing = conn.execute(
            "SELECT id FROM evaluations WHERE prompt_text = ? AND verdict = ? LIMIT 1",
            (prompt, verdict)
        ).fetchone()

        if existing:
            skipped += 1
            continue

        # Generate plain English summary
        block_reason = reranker_r if verdict == "INTERVENE" else ""
        summary = _generate_summary(
            prompt_text=prompt,
            verdict=verdict,
            triggered_on=triggered,
            matched_failure_mode=mode,
            matched_category=category,
            top_risky_similarity=risky_sim,
            reranker_fired=reranked,
            reranker_verdict=reranker_v,
            reranker_reason=reranker_r,
            check_type="pre_check",
            timestamp=row_ts,
        )

        reg_flags = _detect_regulatory_flags(
            matched_category=category,
            matched_failure_mode=mode,
            q1="", q2="", q3="", q4="", q6=reranker_q6,
        )

        try:
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO evaluations
                (timestamp, session_id, agent_id, environment,
                 prompt_text, agent_response_text, check_type,
                 verdict, triggered_on, action_taken, block_reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                row_ts, batch, "failguard", "test",
                prompt, grok_resp, "pre_check",
                verdict, triggered,
                "blocked" if verdict == "INTERVENE" else "passed",
                block_reason,
            ))
            evaluation_id = cur.lastrowid

            cur.execute("""
                INSERT INTO embedding_signals
                (evaluation_id, risky_votes, benign_votes,
                 top_risky_similarity, top_benign_similarity,
                 matched_failure_mode, matched_category)
                VALUES (?,?,?,?,?,?,?)
            """, (evaluation_id, rv, bv, risky_sim, benign_sim, mode, category))

            cur.execute("""
                INSERT INTO reranker_decisions
                (evaluation_id, reranker_fired, reranker_verdict,
                 q1_data_exfiltration, q2_deception, q3_unauthorized_action,
                 q4_consent_violation, q5_safe_verified, q6_bulk_data_internal_doc,
                 reranker_reason, error_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                evaluation_id, 1 if reranked else 0, reranker_v,
                "", "", "", "", "", reranker_q6,
                reranker_r, "",
            ))

            human_review = 1 if verdict == "INTERVENE" and reranked and not correct else 0
            cur.execute("""
                INSERT INTO audit_trail
                (evaluation_id, timestamp, summary, regulatory_flags,
                 human_review_required, human_review_status)
                VALUES (?,?,?,?,?,?)
            """, (
                evaluation_id, row_ts, summary,
                json.dumps(reg_flags),
                human_review,
                "pending" if human_review else "not_required",
            ))

            # Load miss into feedback_footprint if verdict was wrong
            if not correct:
                miss_type      = "false_negative" if label == 1 and verdict == "SAFE" else "false_positive"
                correct_verdict = "INTERVENE" if miss_type == "false_negative" else "SAFE"
                root_cause     = classify_root_cause(row)
                fix_type, fix_description = suggest_fix(root_cause, miss_type, prompt)

                cur.execute("""
                    INSERT INTO feedback_footprint
                    (evaluation_id, identified_at, miss_type, correct_verdict,
                     root_cause, fix_type, fix_description)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    evaluation_id, row_ts,
                    miss_type, correct_verdict,
                    root_cause, fix_type, fix_description,
                ))
                miss_loaded += 1

            conn.commit()
            imported += 1

        except Exception as e:
            conn.rollback()
            print(f"  [ERROR] Row {i+1}: {e}")

    conn.close()

    print(f"\nImport complete:")
    print(f"  Imported  : {imported:,} evaluations")
    print(f"  Skipped   : {skipped:,} (already in database)")
    print(f"  Misses    : {miss_loaded:,} wrong verdicts loaded into feedback_footprint")
def show_summary(db: FailGuardDB):
    """Show overall evaluation statistics."""
    conn = db.get_connection()

    total = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    blocked = conn.execute("SELECT COUNT(*) FROM evaluations WHERE verdict='INTERVENE'").fetchone()[0]
    passed = conn.execute("SELECT COUNT(*) FROM evaluations WHERE verdict='SAFE'").fetchone()[0]
    fp = conn.execute("SELECT COUNT(*) FROM feedback_footprint WHERE miss_type='false_positive'").fetchone()[0]
    fn = conn.execute("SELECT COUNT(*) FROM feedback_footprint WHERE miss_type='false_negative'").fetchone()[0]
    unresolved = conn.execute("SELECT COUNT(*) FROM feedback_footprint WHERE fix_applied=0").fetchone()[0]

    print("\n" + "="*60)
    print("FAILGUARD AUDIT SUMMARY")
    print("="*60)
    print(f"  Total evaluations     : {total:,}")
    print(f"  Actions blocked       : {blocked:,}")
    print(f"  Actions passed        : {passed:,}")
    print(f"  Known false positives : {fp:,}")
    print(f"  Known false negatives : {fn:,}")
    print(f"  Unresolved misses     : {unresolved:,}")

    # Root cause breakdown
    root_causes = conn.execute("""
        SELECT root_cause, COUNT(*) as count
        FROM feedback_footprint
        WHERE fix_applied = 0
        GROUP BY root_cause
        ORDER BY count DESC
    """).fetchall()

    if root_causes:
        print(f"\n  Unresolved miss root causes:")
        for rc in root_causes:
            print(f"    {rc['root_cause']:<35} {rc['count']:>4}")

    conn.close()
    print("="*60)


def show_misses(db: FailGuardDB, miss_type: str = ""):
    """Show all unresolved misses with fix recommendations."""
    conn = db.get_connection()

    query = """
        SELECT f.id, f.miss_type, f.root_cause, f.fix_type, f.fix_description,
               e.prompt_text, e.verdict, e.triggered_on,
               r.reranker_reason, r.reranker_verdict
        FROM feedback_footprint f
        JOIN evaluations e ON f.evaluation_id = e.id
        LEFT JOIN reranker_decisions r ON r.evaluation_id = e.id
        WHERE f.fix_applied = 0
    """
    params = []
    if miss_type:
        query += " AND f.miss_type = ?"
        params.append(miss_type)

    query += " ORDER BY f.miss_type, f.root_cause"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        print("\nNo unresolved misses found.")
        return

    print(f"\n{'='*70}")
    print(f"UNRESOLVED MISSES ({len(rows)})")
    print(f"{'='*70}")

    current_type = None
    for row in rows:
        if row["miss_type"] != current_type:
            current_type = row["miss_type"]
            label = "FALSE POSITIVES (safe actions blocked)" if current_type == "false_positive" else "FALSE NEGATIVES (threats missed)"
            print(f"\n--- {label} ---")

        print(f"\n  [#{row['id']}] Root cause: {row['root_cause']}")
        print(f"  Prompt   : {row['prompt_text'][:80]}")
        print(f"  Verdict  : {row['verdict']}  |  Triggered: {row['triggered_on']}")
        if row["reranker_reason"]:
            print(f"  Reranker : {row['reranker_reason'][:80]}")
        print(f"  Fix type : {row['fix_type']}")
        print(f"  Fix desc : {row['fix_description']}")


def generate_report(db: FailGuardDB, since: str = "", until: str = ""):
    """Generate a plain English compliance report."""
    conn = db.get_connection()

    query = "SELECT * FROM audit_trail a JOIN evaluations e ON a.evaluation_id = e.id"
    params = []
    conditions = []

    if since:
        conditions.append("a.timestamp >= ?")
        params.append(since)
    if until:
        conditions.append("a.timestamp <= ?")
        params.append(until + "T23:59:59")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY a.timestamp"
    rows = conn.execute(query, params).fetchall()

    total = len(rows)
    blocked = sum(1 for r in rows if r["verdict"] == "INTERVENE")
    passed = sum(1 for r in rows if r["verdict"] == "SAFE")
    review_required = sum(1 for r in rows if r["human_review_required"] == 1)

    period = ""
    if since or until:
        period = f" for period {since or 'start'} to {until or 'present'}"

    print(f"\n{'='*70}")
    print(f"FAILGUARD COMPLIANCE REPORT{period.upper()}")
    print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*70}")
    print(f"\nSUMMARY")
    print(f"  Total agent actions evaluated : {total:,}")
    print(f"  Actions blocked (INTERVENE)   : {blocked:,}")
    print(f"  Actions passed (SAFE)         : {passed:,}")
    print(f"  Actions requiring human review: {review_required:,}")

    if total > 0:
        block_rate = blocked / total * 100
        print(f"  Interception rate             : {block_rate:.1f}%")

    print(f"\nBLOCKED ACTIONS")
    blocked_rows = [r for r in rows if r["verdict"] == "INTERVENE"]
    if blocked_rows:
        for row in blocked_rows[:20]:
            print(f"\n  {row['summary']}")
            flags = json.loads(row["regulatory_flags"] or "[]")
            if flags:
                print(f"  Regulatory considerations: {', '.join(flags[:2])}")
        if len(blocked_rows) > 20:
            print(f"\n  ... and {len(blocked_rows) - 20} more blocked actions.")
    else:
        print("  No actions were blocked in this period.")

    print(f"\n{'='*70}")
    print("END OF REPORT")
    print(f"{'='*70}")
    conn.close()


def show_review_queue(db: FailGuardDB):
    """Show all evaluations requiring human review."""
    conn = db.get_connection()
    rows = conn.execute("""
        SELECT a.id, a.timestamp, a.summary, a.human_review_status,
               e.prompt_text, e.verdict
        FROM audit_trail a
        JOIN evaluations e ON a.evaluation_id = e.id
        WHERE a.human_review_required = 1
        ORDER BY a.timestamp DESC
    """).fetchall()
    conn.close()

    if not rows:
        print("\nNo actions currently requiring human review.")
        return

    print(f"\n{'='*60}")
    print(f"HUMAN REVIEW QUEUE ({len(rows)} items)")
    print(f"{'='*60}")
    for row in rows:
        print(f"\n  [{row['id']}] {row['timestamp'][:19]} | Status: {row['human_review_status']}")
        print(f"  {row['summary'][:120]}")


def show_fix_effectiveness(db: FailGuardDB, fix_version: str):
    """Show whether a specific fix version reduced the miss rate."""
    conn = db.get_connection()

    fixed = conn.execute("""
        SELECT COUNT(*) FROM feedback_footprint
        WHERE fix_version = ? AND fix_applied = 1
    """, (fix_version,)).fetchone()[0]

    remaining = conn.execute("""
        SELECT COUNT(*) FROM feedback_footprint
        WHERE fix_applied = 0
    """).fetchone()[0]

    print(f"\n{'='*60}")
    print(f"FIX EFFECTIVENESS: {fix_version}")
    print(f"{'='*60}")
    print(f"  Misses resolved by {fix_version} : {fixed:,}")
    print(f"  Misses still unresolved         : {remaining:,}")
    conn.close()


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FailGuard Miss Analyzer — audit database CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--import-csv", metavar="FILE",
        help="Import ALL rows from a results CSV directly into the database (use for backfill)")
    parser.add_argument("--batch-label", metavar="LABEL",
        help="Optional label for the batch being imported (e.g. 'Batch 1 — Customer Support')")
    parser.add_argument("--csv", metavar="FILE",
        help="Load misses only from a CSV that is already in the database")
    parser.add_argument("--summary", action="store_true",
        help="Show overall evaluation statistics")
    parser.add_argument("--misses", action="store_true",
        help="Show all unresolved misses with fix recommendations")
    parser.add_argument("--fp", action="store_true",
        help="Show false positives only")
    parser.add_argument("--fn", action="store_true",
        help="Show false negatives only")
    parser.add_argument("--report", action="store_true",
        help="Generate a plain English compliance report")
    parser.add_argument("--since", metavar="DATE",
        help="Filter report from date (YYYY-MM-DD)")
    parser.add_argument("--until", metavar="DATE",
        help="Filter report until date (YYYY-MM-DD)")
    parser.add_argument("--review-queue", action="store_true",
        help="Show all evaluations requiring human review")
    parser.add_argument("--mark-fixed", metavar="ID", type=int,
        help="Mark a feedback_footprint entry as fixed")
    parser.add_argument("--fix-version", metavar="VERSION",
        help="Version string for --mark-fixed or --fix-effectiveness")
    parser.add_argument("--fix-effectiveness", metavar="VERSION",
        help="Show whether a fix version reduced the miss rate")
    parser.add_argument("--db", metavar="PATH",
        help="Path to the SQLite database (default: data/failguard_audit.db)")

    args = parser.parse_args()

    db = FailGuardDB(db_path=args.db or "")

    if args.import_csv:
        import_csv(args.import_csv, db, batch_label=args.batch_label or "")

    if args.csv:
        load_csv(args.csv, db)

    if args.summary:
        show_summary(db)

    if args.misses:
        show_misses(db)
    elif args.fp:
        show_misses(db, miss_type="false_positive")
    elif args.fn:
        show_misses(db, miss_type="false_negative")

    if args.report:
        generate_report(db, since=args.since or "", until=args.until or "")

    if args.review_queue:
        show_review_queue(db)

    if args.mark_fixed:
        if not args.fix_version:
            print("ERROR: --mark-fixed requires --fix-version")
            sys.exit(1)
        db.mark_fix_applied(args.mark_fixed, args.fix_version)
        print(f"Marked feedback #{args.mark_fixed} as fixed by {args.fix_version}")

    if args.fix_effectiveness:
        show_fix_effectiveness(db, args.fix_effectiveness)

    if not any([args.import_csv, args.csv, args.summary, args.misses, args.fp, args.fn,
                args.report, args.review_queue, args.mark_fixed, args.fix_effectiveness]):
        parser.print_help()


if __name__ == "__main__":
    main()
