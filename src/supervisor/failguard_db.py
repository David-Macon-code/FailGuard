"""
FailGuard Database Module
=========================

Manages the SQLite audit database for FailGuard evaluations.
Provides async write capability so database operations never
block the main evaluation pipeline.

Database location: data/failguard_audit.db
(relative to the FailGuard project root)

Tables:
  evaluations        — one row per agent action evaluated
  embedding_signals  — embedding layer evidence
  reranker_decisions — LLM reranker 6-question findings
  audit_trail        — plain English compliance record
  feedback_footprint — miss tracking and fix history

Usage:
  from src.supervisor.failguard_db import FailGuardDB
  db = FailGuardDB()
  db.write_async(evaluation_result, prompt, agent_response, check_type, agent_id)
"""

import os
import sqlite3
import threading
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any


# ---------------------------------------------------------------
# Database path — relative to project root
# ---------------------------------------------------------------
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "failguard_audit.db"
)


# ---------------------------------------------------------------
# Schema
# ---------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    session_id          TEXT,
    agent_id            TEXT,
    environment         TEXT DEFAULT 'production',
    prompt_text         TEXT NOT NULL,
    agent_response_text TEXT,
    check_type          TEXT NOT NULL,
    verdict             TEXT NOT NULL,
    triggered_on        TEXT,
    action_taken        TEXT NOT NULL,
    block_reason        TEXT
);

CREATE TABLE IF NOT EXISTS embedding_signals (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id         INTEGER NOT NULL REFERENCES evaluations(id),
    risky_votes           INTEGER,
    benign_votes          INTEGER,
    top_risky_similarity  REAL,
    top_benign_similarity REAL,
    matched_failure_mode  TEXT,
    matched_category      TEXT
);

CREATE TABLE IF NOT EXISTS reranker_decisions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id         INTEGER NOT NULL REFERENCES evaluations(id),
    reranker_fired        INTEGER DEFAULT 0,
    reranker_verdict      TEXT,
    q1_data_exfiltration  TEXT,
    q2_deception          TEXT,
    q3_unauthorized_action TEXT,
    q4_consent_violation  TEXT,
    q5_safe_verified      TEXT,
    q6_bulk_data_internal_doc TEXT,
    reranker_reason       TEXT,
    error_type            TEXT
);

CREATE TABLE IF NOT EXISTS audit_trail (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id         INTEGER NOT NULL REFERENCES evaluations(id),
    timestamp             TEXT NOT NULL,
    summary               TEXT NOT NULL,
    regulatory_flags      TEXT,
    human_review_required INTEGER DEFAULT 0,
    human_review_status   TEXT DEFAULT 'not_required',
    reviewer_notes        TEXT,
    review_timestamp      TEXT
);

CREATE TABLE IF NOT EXISTS feedback_footprint (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id     INTEGER NOT NULL REFERENCES evaluations(id),
    identified_at     TEXT NOT NULL,
    miss_type         TEXT NOT NULL,
    correct_verdict   TEXT NOT NULL,
    root_cause        TEXT,
    fix_type          TEXT,
    fix_description   TEXT,
    fix_applied       INTEGER DEFAULT 0,
    fix_version       TEXT,
    fix_timestamp     TEXT
);

CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp ON evaluations(timestamp);
CREATE INDEX IF NOT EXISTS idx_evaluations_verdict ON evaluations(verdict);
CREATE INDEX IF NOT EXISTS idx_evaluations_agent_id ON evaluations(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_trail_evaluation_id ON audit_trail(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_feedback_miss_type ON feedback_footprint(miss_type);
CREATE INDEX IF NOT EXISTS idx_feedback_fix_applied ON feedback_footprint(fix_applied);
"""


# ---------------------------------------------------------------
# Plain English summary generator
# ---------------------------------------------------------------
def _generate_summary(
    prompt_text: str,
    verdict: str,
    triggered_on: str,
    matched_failure_mode: str,
    matched_category: str,
    top_risky_similarity: float,
    reranker_fired: bool,
    reranker_verdict: str,
    reranker_reason: str,
    check_type: str,
    timestamp: str,
) -> str:
    """
    Generates a plain English audit summary readable by a
    regulator, compliance officer, or court.
    """
    ts = timestamp[:19].replace("T", " ")
    prompt_short = prompt_text[:120] + "..." if len(prompt_text) > 120 else prompt_text

    if verdict == "INTERVENE":
        if reranker_fired and reranker_reason:
            finding = reranker_reason
        else:
            finding = (
                f"it matched the '{matched_failure_mode}' failure pattern "
                f"({matched_category}) with {top_risky_similarity:.1%} confidence"
            )

        trigger_desc = {
            "user_prompt": "The user's request was intercepted before the agent was called.",
            "agent_response": "The agent's proposed response was intercepted before it reached the user.",
            "both": "Both the user's request and the agent's proposed response were intercepted.",
            "reranker": "The safety reranker identified the risk after the embedding layer was inconclusive.",
        }.get(triggered_on, "The action was intercepted.")

        return (
            f"On {ts} UTC, an agent action was evaluated and BLOCKED. "
            f"The action was: '{prompt_short}'. "
            f"{trigger_desc} "
            f"The action was blocked because {finding}. "
            f"No harmful action was executed. No data was transmitted."
        )
    else:
        return (
            f"On {ts} UTC, an agent action was evaluated and APPROVED. "
            f"The action was: '{prompt_short}'. "
            f"The action matched no failure patterns above the safety threshold "
            f"and was permitted to proceed."
        )


# ---------------------------------------------------------------
# Regulatory flag detector
# ---------------------------------------------------------------
def _detect_regulatory_flags(
    matched_category: str,
    matched_failure_mode: str,
    q1: str,
    q2: str,
    q3: str,
    q4: str,
    q6: str,
) -> list:
    """
    Returns a list of potentially applicable regulations based
    on the nature of the intercepted action.
    """
    flags = []

    if q1 == "YES" or q4 == "YES":
        flags.append("GDPR Article 5 (data processing principles)")
        flags.append("CCPA Section 1798.100 (consumer data rights)")
        flags.append("Colorado AI Act (data governance)")

    if q3 == "YES":
        flags.append("Colorado AI Act Section 6-1-1703 (human oversight)")
        flags.append("California ADMT (meaningful human review)")

    if q2 == "YES":
        flags.append("FTC Section 5 (unfair or deceptive acts)")
        flags.append("Colorado AI Act (transparency requirements)")

    if q6 == "YES":
        flags.append("GDPR Article 32 (security of processing)")
        flags.append("CCPA (data minimization)")

    if "Legal and Compliance" in matched_category:
        flags.append("EU AI Act (high-risk AI obligations)")
        flags.append("Colorado AI Act (impact assessment required)")

    return flags


# ---------------------------------------------------------------
# FailGuardDB class
# ---------------------------------------------------------------
class FailGuardDB:
    """
    Manages the FailGuard SQLite audit database.
    All writes are async — they never block the evaluation pipeline.
    """

    def __init__(self, db_path: str = ""):
        self.db_path = db_path or os.path.normpath(_DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        print(f"(OK) FailGuard audit database: {self.db_path}")

    def _init_db(self):
        """Create tables if they do not exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _write_sync(
        self,
        prompt_text: str,
        agent_response_text: str,
        check_type: str,
        verdict: str,
        triggered_on: str,
        action_taken: str,
        block_reason: str,
        risky_votes: int,
        benign_votes: int,
        top_risky_similarity: float,
        top_benign_similarity: float,
        matched_failure_mode: str,
        matched_category: str,
        reranker_fired: bool,
        reranker_verdict: str,
        q1: str,
        q2: str,
        q3: str,
        q4: str,
        q5: str,
        q6: str,
        reranker_reason: str,
        error_type: str,
        session_id: str,
        agent_id: str,
        environment: str,
    ):
        """Synchronous write — called from background thread."""
        timestamp = datetime.now(timezone.utc).isoformat()

        summary = _generate_summary(
            prompt_text=prompt_text,
            verdict=verdict,
            triggered_on=triggered_on,
            matched_failure_mode=matched_failure_mode,
            matched_category=matched_category,
            top_risky_similarity=top_risky_similarity,
            reranker_fired=reranker_fired,
            reranker_verdict=reranker_verdict,
            reranker_reason=reranker_reason,
            check_type=check_type,
            timestamp=timestamp,
        )

        reg_flags = _detect_regulatory_flags(
            matched_category=matched_category,
            matched_failure_mode=matched_failure_mode,
            q1=q1, q2=q2, q3=q3, q4=q4, q6=q6,
        )

        human_review_required = 1 if verdict == "INTERVENE" and reranker_fired and error_type else 0

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO evaluations
                (timestamp, session_id, agent_id, environment,
                 prompt_text, agent_response_text, check_type,
                 verdict, triggered_on, action_taken, block_reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                timestamp, session_id, agent_id, environment,
                prompt_text, agent_response_text, check_type,
                verdict, triggered_on, action_taken, block_reason,
            ))
            evaluation_id = cur.lastrowid

            cur.execute("""
                INSERT INTO embedding_signals
                (evaluation_id, risky_votes, benign_votes,
                 top_risky_similarity, top_benign_similarity,
                 matched_failure_mode, matched_category)
                VALUES (?,?,?,?,?,?,?)
            """, (
                evaluation_id, risky_votes, benign_votes,
                top_risky_similarity, top_benign_similarity,
                matched_failure_mode, matched_category,
            ))

            cur.execute("""
                INSERT INTO reranker_decisions
                (evaluation_id, reranker_fired, reranker_verdict,
                 q1_data_exfiltration, q2_deception, q3_unauthorized_action,
                 q4_consent_violation, q5_safe_verified, q6_bulk_data_internal_doc,
                 reranker_reason, error_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                evaluation_id, 1 if reranker_fired else 0, reranker_verdict,
                q1, q2, q3, q4, q5, q6,
                reranker_reason, error_type,
            ))

            cur.execute("""
                INSERT INTO audit_trail
                (evaluation_id, timestamp, summary, regulatory_flags,
                 human_review_required, human_review_status)
                VALUES (?,?,?,?,?,?)
            """, (
                evaluation_id, timestamp, summary,
                json.dumps(reg_flags),
                human_review_required,
                "pending" if human_review_required else "not_required",
            ))

            conn.commit()

        except Exception as e:
            print(f"[FailGuardDB] Write error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def write_async(
        self,
        result,
        prompt_text: str,
        agent_response_text: str = "",
        check_type: str = "pre_check",
        session_id: str = "",
        agent_id: str = "default",
        environment: str = "production",
    ):
        """
        Async write — fires a background thread and returns immediately.
        The evaluation pipeline is never blocked.

        Args:
            result:             EvaluationResult from FailGuardSupervisor
            prompt_text:        The user prompt or action evaluated
            agent_response_text: The agent response (post-check only)
            check_type:         'pre_check' or 'post_check'
            session_id:         Optional session identifier
            agent_id:           Which agent is being supervised
            environment:        'production', 'staging', or 'test'
        """
        kwargs = dict(
            prompt_text=prompt_text,
            agent_response_text=agent_response_text,
            check_type=check_type,
            verdict=result.status,
            triggered_on=result.triggered_on,
            action_taken="blocked" if result.status == "INTERVENE" else "passed",
            block_reason=result.reranker_reason if result.status == "INTERVENE" else "",
            risky_votes=result.risky_vote_count,
            benign_votes=result.benign_vote_count,
            top_risky_similarity=result.similarity_score,
            top_benign_similarity=result.benign_similarity_score,
            matched_failure_mode=result.matched_mode,
            matched_category=result.matched_category,
            reranker_fired=result.reranked,
            reranker_verdict=result.reranker_verdict,
            q1=getattr(result, "reranker_q1", ""),
            q2=getattr(result, "reranker_q2", ""),
            q3=getattr(result, "reranker_q3", ""),
            q4=getattr(result, "reranker_q4", ""),
            q5=getattr(result, "reranker_q5", ""),
            q6=result.reranker_q6,
            reranker_reason=result.reranker_reason,
            error_type="",
            session_id=session_id,
            agent_id=agent_id,
            environment=environment,
        )

        thread = threading.Thread(
            target=self._write_sync,
            kwargs=kwargs,
            daemon=True,
        )
        thread.start()

    def log_miss(
        self,
        evaluation_id: int,
        miss_type: str,
        correct_verdict: str,
        root_cause: str = "",
        fix_type: str = "",
        fix_description: str = "",
    ):
        """
        Record a missed verdict in the feedback_footprint table.
        Called by the Miss Analyzer when a wrong verdict is identified.

        Args:
            evaluation_id:   The evaluation that was wrong
            miss_type:       'false_negative' or 'false_positive'
            correct_verdict: 'INTERVENE' or 'SAFE'
            root_cause:      e.g. 'embedding_level', 'reranker_q5', 'benign_votes_won'
            fix_type:        e.g. 'benign_anchor', 'rubric_change', 'taxonomy_mode'
            fix_description: Plain English description of the fix needed
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO feedback_footprint
                (evaluation_id, identified_at, miss_type, correct_verdict,
                 root_cause, fix_type, fix_description)
                VALUES (?,?,?,?,?,?,?)
            """, (
                evaluation_id,
                datetime.now(timezone.utc).isoformat(),
                miss_type,
                correct_verdict,
                root_cause,
                fix_type,
                fix_description,
            ))
            conn.commit()
        finally:
            conn.close()

    def mark_fix_applied(
        self,
        feedback_id: int,
        fix_version: str,
        fix_description: str = "",
    ):
        """
        Mark a feedback_footprint entry as fixed.
        Called after a new reranker or supervisor version is deployed.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE feedback_footprint
                SET fix_applied = 1,
                    fix_version = ?,
                    fix_description = COALESCE(NULLIF(?, ''), fix_description),
                    fix_timestamp = ?
                WHERE id = ?
            """, (
                fix_version,
                fix_description,
                datetime.now(timezone.utc).isoformat(),
                feedback_id,
            ))
            conn.commit()
        finally:
            conn.close()

    def get_connection(self) -> sqlite3.Connection:
        """Return a read-only connection for the Miss Analyzer."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
