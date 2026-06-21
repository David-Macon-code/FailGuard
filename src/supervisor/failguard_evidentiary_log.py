"""
failguard_evidentiary_log.py

Hash-chained, digitally-signed, append-only audit logging for FailGuard.

DESIGN GOALS (per architecture discussion, June 2026)
------------------------------------------------------
1. ONE log serves both purposes: enterprise data-mining AND legally-defensible
   evidence. There is no separate "encrypted" log — encryption protects
   confidentiality, not integrity, and was the wrong tool for this job.

2. Integrity (tamper-evidence) is provided by a SHA-256 hash chain: each
   entry's hash incorporates the previous entry's hash, so any retroactive
   edit to any historical entry breaks the chain for every entry after it.
   Verifying this requires NO secret key — anyone with the log file can
   recompute the chain and check it.

3. Authenticity is provided by a digital signature (Ed25519) computed with a
   PRIVATE KEY GENERATED LOCALLY, on the machine/deployment actually running
   FailGuard. This key is never transmitted, never held centrally by
   BITLJAYWBS LLC / FailGuard the vendor, and never leaves the enterprise's
   own environment. The PUBLIC half is meant to be exported and given to
   auditors, regulators, or counsel so they can verify authenticity
   independently, without needing FailGuard (the company) to still exist,
   cooperate, or be trusted.

4. The canonical log file must remain APPEND-ONLY. Anyone wanting to
   manipulate/clean/reshape data for analytics should export a derived copy
   rather than edit this file directly -- editing the canonical file breaks
   the hash chain and destroys its evidentiary value.

5. This module deliberately does NOT implement encryption-at-rest or
   external (e.g. OpenTimestamps/blockchain) anchoring. Those are optional,
   separable additions -- see the "NEXT STEPS" note at the bottom of this
   file -- and should not be conflated with the integrity/authenticity
   guarantees this module already provides on its own.

FILE FORMAT
-----------
CSV, UTF-8, one row per evaluation. All existing FailGuard business fields
(see BUSINESS_FIELDNAMES below) are preserved exactly as-is for drop-in
compatibility with existing data-mining workflows (pandas, Excel, BI tools).
Three additional columns are appended:

    timestamp_utc   ISO-8601 UTC timestamp, set at the moment of writing
    prev_hash       hex SHA-256 hash of the previous row (or GENESIS_HASH
                     for the first row in a given log file)
    entry_hash      hex SHA-256 hash of this row's canonical content
                     (business fields + timestamp_utc + prev_hash)
    signature       hex Ed25519 signature of entry_hash, base16-encoded

Analysts using the log for data-mining can simply ignore the last four
columns. Auditors/courts use verify_log() to check them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GENESIS_HASH = "0" * 64  # prev_hash value used for the first entry in a log

# The existing FailGuard CSV business fields (from langgraph_protected_agent_v9.py),
# preserved exactly so this module is a drop-in replacement for the existing
# csv.DictWriter call, not a breaking schema change.
BUSINESS_FIELDNAMES = [
    "idx", "prompt", "label", "label_str", "status", "correct",
    "similarity", "benign_sim", "risky_votes", "benign_votes",
    "confidence", "matched_mode", "matched_category", "triggered_on",
    "reranked", "reranker_verdict", "reranker_reason", "reranker_q6",
    "grok_invoked", "grok_response", "explanation",
]

INTEGRITY_FIELDNAMES = ["timestamp_utc", "prev_hash", "entry_hash", "signature"]
ALL_FIELDNAMES = BUSINESS_FIELDNAMES + INTEGRITY_FIELDNAMES

DEFAULT_KEY_DIR = os.path.join("data", "keys")
DEFAULT_PRIVATE_KEY_PATH = os.path.join(DEFAULT_KEY_DIR, "failguard_signing_key.pem")
DEFAULT_PUBLIC_KEY_PATH = os.path.join(DEFAULT_KEY_DIR, "failguard_signing_key.pub")


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def load_or_create_keypair(
    private_key_path: str = DEFAULT_PRIVATE_KEY_PATH,
    public_key_path: str = DEFAULT_PUBLIC_KEY_PATH,
) -> Ed25519PrivateKey:
    """
    Loads this deployment's local Ed25519 signing key, generating one on
    first run if it doesn't exist yet.

    IMPORTANT: this key is generated and stored LOCALLY. It should never be
    copied off this machine, committed to source control, or transmitted to
    FailGuard the vendor. Restrict filesystem permissions on the directory
    it lives in. The public key (the .pub file) is safe and intended to be
    shared with auditors/regulators/counsel.
    """
    os.makedirs(os.path.dirname(private_key_path), exist_ok=True)

    if os.path.exists(private_key_path):
        with open(private_key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(f"Key at {private_key_path} is not an Ed25519 key")
        return private_key

    # First run on this deployment: generate a new keypair.
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(private_key_path, "wb") as f:
        f.write(private_bytes)
    try:
        os.chmod(private_key_path, 0o600)  # owner read/write only, best-effort
    except OSError:
        pass  # platform may not support chmod (e.g., some Windows setups)

    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with open(public_key_path, "wb") as f:
        f.write(public_bytes)

    print(f"[EvidentiaryLog] New local signing key generated: {private_key_path}")
    print(f"[EvidentiaryLog] Public key for auditors/counsel: {public_key_path}")
    print("[EvidentiaryLog] The private key never leaves this machine. Do not commit it to git.")

    return private_key


def load_public_key(public_key_path: str = DEFAULT_PUBLIC_KEY_PATH) -> Ed25519PublicKey:
    with open(public_key_path, "rb") as f:
        key = serialization.load_pem_public_key(f.read())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError(f"Key at {public_key_path} is not an Ed25519 public key")
    return key


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _canonical_bytes(row: Dict[str, Any]) -> bytes:
    """
    Deterministic serialization of a row's content for hashing. Sorted keys,
    no extraneous whitespace, so the same logical content always produces
    the same hash regardless of dict ordering.
    """
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def compute_entry_hash(business_row: Dict[str, Any], timestamp_utc: str, prev_hash: str) -> str:
    payload = dict(business_row)
    payload["timestamp_utc"] = timestamp_utc
    payload["prev_hash"] = prev_hash
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class EvidentiaryLogWriter:
    """
    Append-only, hash-chained, signed CSV log writer.

    Usage:
        writer = EvidentiaryLogWriter("logs/results_v9_20260619.csv")
        writer.append({"idx": 0, "prompt": "...", "status": "SAFE", ...})
        writer.append({"idx": 1, "prompt": "...", "status": "INTERVENE", ...})
        writer.close()

    Safe to stop and resume: if the target file already exists, the writer
    reads the last row to recover the chain's current head hash before
    appending further entries, so a process restart does not break the
    chain or require starting a new file.
    """

    def __init__(
        self,
        csv_path: str,
        private_key_path: str = DEFAULT_PRIVATE_KEY_PATH,
        public_key_path: str = DEFAULT_PUBLIC_KEY_PATH,
    ):
        self.csv_path = csv_path
        self._private_key = load_or_create_keypair(private_key_path, public_key_path)
        self._head_hash = self._recover_head_hash()
        self._file_exists = os.path.exists(csv_path)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    def _recover_head_hash(self) -> str:
        if not os.path.exists(self.csv_path):
            return GENESIS_HASH
        last_hash = GENESIS_HASH
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_hash = row.get("entry_hash", GENESIS_HASH)
        return last_hash

    def append(self, business_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appends one evaluation record to the log. Returns the full row
        (business fields + integrity fields) as written, in case the caller
        wants to log/display it.
        """
        # CSV round-trips everything as strings (csv.DictReader never returns
        # bool/int/float -- only str). The hash MUST be computed over the
        # same representation that will later be read back during
        # verification, or verification will spuriously fail on every row.
        # So: stringify here, at write time, matching what csv.DictWriter
        # would itself produce -- not the original Python-typed values.
        def _stringify(v: Any) -> str:
            return "" if v is None else str(v)

        normalized = {k: _stringify(business_row.get(k, "")) for k in BUSINESS_FIELDNAMES}

        timestamp_utc = datetime.now(timezone.utc).isoformat()
        entry_hash = compute_entry_hash(normalized, timestamp_utc, self._head_hash)
        signature = self._private_key.sign(bytes.fromhex(entry_hash)).hex()

        full_row = dict(normalized)
        full_row["timestamp_utc"] = timestamp_utc
        full_row["prev_hash"] = self._head_hash
        full_row["entry_hash"] = entry_hash
        full_row["signature"] = signature

        write_header = not self._file_exists
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(full_row)
            f.flush()
            os.fsync(f.fileno())  # durability: survive a crash immediately after write

        self._file_exists = True
        self._head_hash = entry_hash
        return full_row

    def append_many(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self.append(row)

    def close(self) -> None:
        pass  # no persistent file handle is held open between appends; present for API symmetry


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    ok: bool
    rows_checked: int
    first_failure_row: Optional[int] = None
    failure_reason: Optional[str] = None
    details: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.ok:
            return f"PASS — {self.rows_checked} entries verified, chain intact, all signatures valid."
        return (
            f"FAIL — tampering or corruption detected at row {self.first_failure_row} "
            f"({self.rows_checked} rows checked before failure).\nReason: {self.failure_reason}"
        )


def verify_log(
    csv_path: str,
    public_key_path: str = DEFAULT_PUBLIC_KEY_PATH,
) -> VerificationResult:
    """
    Independently verifies an evidentiary log file:
      1. Recomputes each row's hash from its business content and checks it
         matches the stored entry_hash (detects content tampering).
      2. Checks each row's prev_hash matches the previous row's entry_hash
         (detects row deletion, insertion, or reordering).
      3. Verifies each row's signature against the public key (detects
         forged/unsigned rows).

    Requires only the public key -- no secret, no access to FailGuard, no
    cooperation from the vendor or original logging process required.
    """
    public_key = load_public_key(public_key_path)

    if not os.path.exists(csv_path):
        return VerificationResult(ok=False, rows_checked=0, first_failure_row=0,
                                   failure_reason=f"File not found: {csv_path}")

    expected_prev_hash = GENESIS_HASH
    rows_checked = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            rows_checked = i
            business_row = {k: row.get(k, "") for k in BUSINESS_FIELDNAMES}
            timestamp_utc = row.get("timestamp_utc", "")
            prev_hash = row.get("prev_hash", "")
            entry_hash = row.get("entry_hash", "")
            signature_hex = row.get("signature", "")

            # 1. Chain continuity
            if prev_hash != expected_prev_hash:
                return VerificationResult(
                    ok=False, rows_checked=rows_checked, first_failure_row=i,
                    failure_reason=(
                        f"prev_hash mismatch: row claims prev_hash={prev_hash[:16]}..., "
                        f"but the actual previous entry's hash was {expected_prev_hash[:16]}.... "
                        "This indicates a row was inserted, deleted, or reordered."
                    ),
                )

            # 2. Content integrity
            recomputed_hash = compute_entry_hash(business_row, timestamp_utc, prev_hash)
            if recomputed_hash != entry_hash:
                return VerificationResult(
                    ok=False, rows_checked=rows_checked, first_failure_row=i,
                    failure_reason=(
                        f"entry_hash mismatch: recomputed={recomputed_hash[:16]}..., "
                        f"stored={entry_hash[:16]}.... "
                        "This indicates the content of this row was edited after it was written."
                    ),
                )

            # 3. Signature authenticity
            try:
                public_key.verify(bytes.fromhex(signature_hex), bytes.fromhex(entry_hash))
            except (InvalidSignature, ValueError) as e:
                return VerificationResult(
                    ok=False, rows_checked=rows_checked, first_failure_row=i,
                    failure_reason=(
                        f"signature invalid: {e}. This indicates the row's signature does not "
                        "match the provided public key -- either forged, corrupted, or signed "
                        "with a different key."
                    ),
                )

            expected_prev_hash = entry_hash

    return VerificationResult(ok=True, rows_checked=rows_checked)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FailGuard evidentiary log tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="Verify an evidentiary log file's integrity and authenticity")
    p_verify.add_argument("csv_path", help="Path to the log CSV to verify")
    p_verify.add_argument("--pubkey", default=DEFAULT_PUBLIC_KEY_PATH, help="Path to the signer's public key")

    p_genkey = sub.add_parser("init-key", help="Generate a local signing keypair if one doesn't exist yet")
    p_genkey.add_argument("--private", default=DEFAULT_PRIVATE_KEY_PATH)
    p_genkey.add_argument("--public", default=DEFAULT_PUBLIC_KEY_PATH)

    args = parser.parse_args()

    if args.command == "verify":
        result = verify_log(args.csv_path, args.pubkey)
        print(result)
        raise SystemExit(0 if result.ok else 1)

    elif args.command == "init-key":
        load_or_create_keypair(args.private, args.public)


# ---------------------------------------------------------------------------
# NEXT STEPS (not implemented in this module -- see conversation notes)
# ---------------------------------------------------------------------------
# - External anchoring (e.g. OpenTimestamps) of periodic chain-head hashes,
#   to make the chain's history provably unalterable even against an
#   attacker with full filesystem access to the enterprise's own machine.
# - Optional encryption-at-rest layer, if confidentiality (not just
#   integrity) of log contents is also required -- a SEPARATE concern from
#   everything in this module, to be layered on top, not mixed in.
