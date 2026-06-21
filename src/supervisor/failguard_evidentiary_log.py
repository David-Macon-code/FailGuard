"""
failguard_evidentiary_log.py

Hash-chained, digitally-signed, append-only audit logging for FailGuard.

DESIGN GOALS (per architecture discussion, June 2026)
------------------------------------------------------
1. ONE log serves both purposes: enterprise data-mining AND legally-defensible
   evidence. There is no separate "encrypted" log -- encryption protects
   confidentiality, not integrity, and was the wrong tool for this job.

2. Integrity (tamper-evidence) is provided by a SHA-256 hash chain: each
   entry's hash incorporates the previous entry's hash, so any retroactive
   edit to any historical entry breaks the chain for every entry after it.
   Verifying this requires NO secret key -- anyone with the log file can
   recompute the chain and check it.

3. Authenticity is provided by a digital signature computed with a PRIVATE
   KEY GENERATED LOCALLY, on the machine/deployment actually running
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

5. SIGNATURE SCHEME IS PLUGGABLE (the "PQC seam"). The signing algorithm is
   abstracted behind SignatureScheme rather than hardcoded throughout the
   module. The default and only implementation today is Ed25519 -- fast,
   widely supported, NOT post-quantum-resistant. When NIST finalizes a
   post-quantum signature standard (HAWK or otherwise -- as of this writing,
   HAWK has only advanced to Round 3 of NIST's evaluation, announced May
   2026; nothing in that family is standardized or production-ready yet),
   migrating means writing one new SignatureScheme implementation, not
   touching the hash-chain or verification logic.

   Because this is an APPEND-ONLY log, a future migration can never rewrite
   history -- so every entry records which algorithm signed it
   (`sig_algorithm`), and that field is itself part of the hashed payload
   (tampering with the recorded algorithm breaks the chain like any other
   edit). A single log can therefore correctly verify entries signed under
   different schemes across its lifetime, which is the only honest way to
   support algorithm migration on a log that can't be retroactively edited.

6. This module deliberately does NOT implement encryption-at-rest or
   external (e.g. OpenTimestamps/blockchain) anchoring. Those are optional,
   separable additions -- see the "NEXT STEPS" note at the bottom of this
   file.

FILE FORMAT
-----------
CSV, UTF-8, one row per evaluation. All existing FailGuard business fields
(see BUSINESS_FIELDNAMES below) are preserved exactly as-is for drop-in
compatibility with existing data-mining workflows (pandas, Excel, BI tools).
Five additional columns are appended:

    timestamp_utc   ISO-8601 UTC timestamp, set at the moment of writing
    prev_hash       hex SHA-256 hash of the previous row (or GENESIS_HASH
                     for the first row in a given log file)
    entry_hash      hex SHA-256 hash of this row's canonical content
                     (business fields + timestamp_utc + prev_hash + sig_algorithm)
    sig_algorithm   short identifier of the signature scheme used for this
                     row, e.g. "ed25519" -- itself part of the hashed payload
    signature       hex signature of entry_hash, produced by the scheme
                     named in sig_algorithm

Analysts using the log for data-mining can simply ignore the last five
columns. Auditors/courts use verify_log() to check them.

SCHEMA NOTE: this version changes the hash payload (adds sig_algorithm to
what's hashed) compared to the very first version of this module. This is a
one-time breaking change, made deliberately while real evidentiary logs are
still new/low-volume, specifically so a payload change never has to happen
again after a log is actually relied upon. Pre-existing log files written
before this change will NOT verify under this version -- start a fresh log
file rather than trying to reconcile the two formats.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import sys
import threading
import traceback
from abc import ABC, abstractmethod
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

INTEGRITY_FIELDNAMES = ["timestamp_utc", "prev_hash", "entry_hash", "sig_algorithm", "signature"]
ALL_FIELDNAMES = BUSINESS_FIELDNAMES + INTEGRITY_FIELDNAMES

DEFAULT_KEY_DIR = os.path.join("data", "keys")
# Unsuffixed paths, preserved exactly as the original (pre-multi-scheme)
# names so any key already generated under the original version of this
# module keeps working without regeneration. Only non-default schemes get
# algorithm-suffixed paths -- see _default_key_paths() below.
DEFAULT_PRIVATE_KEY_PATH = os.path.join(DEFAULT_KEY_DIR, "failguard_signing_key.pem")
DEFAULT_PUBLIC_KEY_PATH = os.path.join(DEFAULT_KEY_DIR, "failguard_signing_key.pub")


# ---------------------------------------------------------------------------
# Signature scheme abstraction (the "PQC seam")
# ---------------------------------------------------------------------------

class SignatureScheme(ABC):
    """
    Abstraction over a signing algorithm. EvidentiaryLogWriter and verify_log()
    are written against this interface, not against any specific algorithm,
    so a future scheme (post-quantum or otherwise) can be added by writing
    one new subclass -- no changes needed to the hash-chain or verification
    control flow.
    """

    algorithm_id: str  # short, stable identifier stored per-row, e.g. "ed25519"

    @abstractmethod
    def generate_private_key(self): ...

    @abstractmethod
    def serialize_private_key(self, key) -> bytes: ...

    @abstractmethod
    def deserialize_private_key(self, data: bytes): ...

    @abstractmethod
    def public_key_from_private(self, private_key): ...

    @abstractmethod
    def serialize_public_key(self, key) -> bytes: ...

    @abstractmethod
    def deserialize_public_key(self, data: bytes): ...

    @abstractmethod
    def sign(self, private_key, message: bytes) -> bytes: ...

    @abstractmethod
    def verify(self, public_key, signature: bytes, message: bytes) -> None:
        """Must raise (InvalidSignature or similar) on failure, return None on success."""
        ...


class Ed25519Scheme(SignatureScheme):
    """Current default. Fast (microsecond signing), small (64-byte)
    signatures, widely supported. NOT post-quantum-resistant -- a
    sufficiently capable future quantum computer could break it. Chosen
    deliberately for now: the post-quantum candidates evaluated as of this
    writing (Falcon's floating-point implementation risk, Dilithium's
    multi-kilobyte signatures, HAWK's pre-standardization status) are each
    worse fits for a per-row log signature today than waiting for NIST to
    actually finalize one."""

    algorithm_id = "ed25519"

    def generate_private_key(self):
        return Ed25519PrivateKey.generate()

    def serialize_private_key(self, key) -> bytes:
        return key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def deserialize_private_key(self, data: bytes):
        key = serialization.load_pem_private_key(data, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError("Key is not an Ed25519 private key")
        return key

    def public_key_from_private(self, private_key):
        return private_key.public_key()

    def serialize_public_key(self, key) -> bytes:
        return key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def deserialize_public_key(self, data: bytes):
        key = serialization.load_pem_public_key(data)
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("Key is not an Ed25519 public key")
        return key

    def sign(self, private_key, message: bytes) -> bytes:
        return private_key.sign(message)

    def verify(self, public_key, signature: bytes, message: bytes) -> None:
        public_key.verify(signature, message)  # raises InvalidSignature on failure


# Registry of available schemes, keyed by the algorithm_id stored per-row.
# Adding post-quantum support later means: write a new SignatureScheme
# subclass, register it here, and optionally switch DEFAULT_SCHEME -- old
# rows keep verifying under whichever scheme they were actually signed with.
SIGNATURE_SCHEMES: Dict[str, SignatureScheme] = {
    "ed25519": Ed25519Scheme(),
}
DEFAULT_SCHEME: SignatureScheme = SIGNATURE_SCHEMES["ed25519"]


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def _default_key_paths(scheme: SignatureScheme):
    """Backward-compatible default path resolution: the default scheme keeps
    the original, unsuffixed filenames (so existing keys generated before
    multi-scheme support keep working). Any other scheme gets its own
    algorithm-suffixed filenames so it can coexist with the default scheme's
    key without clobbering it."""
    if scheme.algorithm_id == DEFAULT_SCHEME.algorithm_id:
        return DEFAULT_PRIVATE_KEY_PATH, DEFAULT_PUBLIC_KEY_PATH
    priv = os.path.join(DEFAULT_KEY_DIR, f"failguard_signing_key_{scheme.algorithm_id}.pem")
    pub = os.path.join(DEFAULT_KEY_DIR, f"failguard_signing_key_{scheme.algorithm_id}.pub")
    return priv, pub


def load_or_create_keypair(
    private_key_path: Optional[str] = None,
    public_key_path: Optional[str] = None,
    scheme: SignatureScheme = DEFAULT_SCHEME,
):
    """
    Loads this deployment's local signing key under the given scheme,
    generating one on first run if it doesn't exist yet.

    IMPORTANT: this key is generated and stored LOCALLY. It should never be
    copied off this machine, committed to source control, or transmitted to
    FailGuard the vendor. Restrict filesystem permissions on the directory
    it lives in. The public key (the .pub file) is safe and intended to be
    shared with auditors/regulators/counsel.
    """
    if private_key_path is None or public_key_path is None:
        default_priv, default_pub = _default_key_paths(scheme)
        private_key_path = private_key_path or default_priv
        public_key_path = public_key_path or default_pub

    os.makedirs(os.path.dirname(private_key_path), exist_ok=True)

    if os.path.exists(private_key_path):
        with open(private_key_path, "rb") as f:
            return scheme.deserialize_private_key(f.read())

    # First run on this deployment under this scheme: generate a new keypair.
    private_key = scheme.generate_private_key()
    with open(private_key_path, "wb") as f:
        f.write(scheme.serialize_private_key(private_key))
    try:
        os.chmod(private_key_path, 0o600)  # owner read/write only, best-effort
    except OSError:
        pass  # platform may not support chmod (e.g., some Windows setups)

    public_key = scheme.public_key_from_private(private_key)
    with open(public_key_path, "wb") as f:
        f.write(scheme.serialize_public_key(public_key))

    print(f"[EvidentiaryLog] New local {scheme.algorithm_id} signing key generated: {private_key_path}")
    print(f"[EvidentiaryLog] Public key for auditors/counsel: {public_key_path}")
    print("[EvidentiaryLog] The private key never leaves this machine. Do not commit it to git.")

    return private_key


def load_public_key(public_key_path: Optional[str] = None, scheme: SignatureScheme = DEFAULT_SCHEME):
    if public_key_path is None:
        _, public_key_path = _default_key_paths(scheme)
    with open(public_key_path, "rb") as f:
        return scheme.deserialize_public_key(f.read())


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


def compute_entry_hash(
    business_row: Dict[str, Any], timestamp_utc: str, prev_hash: str, sig_algorithm: str,
) -> str:
    """sig_algorithm is part of the hashed payload deliberately -- otherwise
    someone could alter which algorithm a row claims to be signed under
    without the hash chain catching it."""
    payload = dict(business_row)
    payload["timestamp_utc"] = timestamp_utc
    payload["prev_hash"] = prev_hash
    payload["sig_algorithm"] = sig_algorithm
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class EvidentiaryLogWriter:
    """
    Append-only, hash-chained, signed CSV log writer.

    Usage:
        writer = EvidentiaryLogWriter("logs/results.csv")
        writer.append({"idx": 0, "prompt": "...", "status": "SAFE", ...})
        writer.flush()   # optional: block until all queued writes land on disk

    By default, writes are ASYNCHRONOUS: append() drops the row onto an
    in-memory queue and returns immediately (microseconds, no disk I/O, no
    signing -- safe to call from a request hot path under high concurrency).
    A single dedicated background thread consumes the queue strictly in
    FIFO order and performs the actual hashing, signing, write, and fsync.

    This is deliberately NOT "spawn a thread per write" (the pattern used
    elsewhere in this codebase for the SQLite audit DB). The hash chain is
    order-dependent -- each entry's prev_hash must be the exact previous
    entry's entry_hash -- so concurrent writers racing on the chain head
    would silently corrupt the very tamper-evidence guarantee this module
    exists to provide. A single ordered consumer makes that race structurally
    impossible rather than relying on locking discipline to prevent it.

    The signature scheme is pluggable (scheme=). Default is Ed25519. See the
    module docstring for why, and how a future post-quantum migration works.

    For tests or callers that need a write to be durable and verifiable
    before the next line of code runs, pass async_write=False, or call
    flush() after appending.

    Safe to stop and resume: if the target file already exists, the writer
    reads the last row to recover the chain's current head hash before
    appending further entries, so a process restart does not break the
    chain or require starting a new file.
    """

    def __init__(
        self,
        csv_path: str,
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None,
        async_write: bool = True,
        scheme: SignatureScheme = DEFAULT_SCHEME,
    ):
        self.csv_path = csv_path
        self.scheme = scheme
        self._private_key = load_or_create_keypair(private_key_path, public_key_path, scheme=scheme)
        self._head_hash = self._recover_head_hash()
        self._file_exists = os.path.exists(csv_path)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        # Entries that failed to write in the background (e.g. disk full,
        # permission error). Checked via get_failed_writes() -- a silently
        # dropped evidentiary record is a much worse failure mode than a
        # slow one, so these are never just swallowed.
        self._failed_items: List[Dict[str, Any]] = []
        self._failed_lock = threading.Lock()

        self.async_write = async_write
        self._queue: Optional["queue.Queue"] = None
        self._worker: Optional[threading.Thread] = None
        if self.async_write:
            self._queue = queue.Queue()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    def _recover_head_hash(self) -> str:
        if not os.path.exists(self.csv_path):
            return GENESIS_HASH
        last_hash = GENESIS_HASH
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_hash = row.get("entry_hash", GENESIS_HASH)
        return last_hash

    def _worker_loop(self) -> None:
        while True:
            business_row = self._queue.get()
            try:
                self._append_sync(business_row)
            except Exception:
                # Never let a single bad entry kill the worker thread and
                # silently stop all future logging. Surface it loudly and
                # keep going.
                print(
                    f"[EvidentiaryLog] ERROR writing entry in background worker "
                    f"-- this entry was NOT recorded:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                with self._failed_lock:
                    self._failed_items.append(business_row)
            finally:
                self._queue.task_done()

    def _append_sync(self, business_row: Dict[str, Any]) -> Dict[str, Any]:
        """The actual hash/sign/write/fsync work. Runs on the background
        worker thread in async mode, or inline in synchronous mode."""

        # CSV round-trips everything as strings (csv.DictReader never returns
        # bool/int/float -- only str). The hash MUST be computed over the
        # same representation that will later be read back during
        # verification, or verification will spuriously fail on every row.
        def _stringify(v: Any) -> str:
            return "" if v is None else str(v)

        normalized = {k: _stringify(business_row.get(k, "")) for k in BUSINESS_FIELDNAMES}

        timestamp_utc = datetime.now(timezone.utc).isoformat()
        algorithm_id = self.scheme.algorithm_id
        entry_hash = compute_entry_hash(normalized, timestamp_utc, self._head_hash, algorithm_id)
        signature = self.scheme.sign(self._private_key, bytes.fromhex(entry_hash)).hex()

        full_row = dict(normalized)
        full_row["timestamp_utc"] = timestamp_utc
        full_row["prev_hash"] = self._head_hash
        full_row["entry_hash"] = entry_hash
        full_row["sig_algorithm"] = algorithm_id
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

    def append(self, business_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Appends one evaluation record to the log.

        Async mode (default): returns immediately (None) after queuing --
        the row hasn't been hashed/signed/written yet, so there's nothing
        meaningful to return. Use flush() if you need to know the write
        actually landed before proceeding.

        Synchronous mode (async_write=False): blocks until the row is
        hashed, signed, and durably written, and returns the full row
        (business fields + integrity fields) as written.
        """
        if self.async_write:
            self._queue.put(dict(business_row))
            return None
        return self._append_sync(business_row)

    def append_many(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self.append(row)

    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Blocks until every currently-queued entry has been written to disk.
        Call this before process exit, or in tests, whenever you need a
        guarantee that nothing is still sitting in the in-memory queue.
        """
        if self.async_write and self._queue is not None:
            self._queue.join()

    def get_failed_writes(self) -> List[Dict[str, Any]]:
        """Returns any entries that failed to write in the background.
        A non-empty result means evaluations happened that are NOT reflected
        in the evidentiary log -- this should be monitored/alerted on, not
        silently ignored."""
        with self._failed_lock:
            return list(self._failed_items)

    def close(self) -> None:
        if self.async_write:
            self.flush()


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
    public_key_path: Optional[str] = None,
    public_key_paths: Optional[Dict[str, str]] = None,
) -> VerificationResult:
    """
    Independently verifies an evidentiary log file:
      1. Recomputes each row's hash from its business content (including the
         recorded signature algorithm) and checks it matches the stored
         entry_hash (detects content tampering, including tampering with
         which algorithm a row claims to be signed under).
      2. Checks each row's prev_hash matches the previous row's entry_hash
         (detects row deletion, insertion, or reordering).
      3. Verifies each row's signature, using whichever SignatureScheme its
         sig_algorithm column names, against the public key (detects
         forged/unsigned rows).

    Requires only the public key(s) -- no secret, no access to FailGuard, no
    cooperation from the vendor or original logging process required.

    public_key_path: convenience for the common single-scheme case (maps to
    the default scheme's key). public_key_paths: {algorithm_id: path} for
    logs that may contain entries signed under more than one scheme over
    their lifetime (e.g. mid-migration to a new algorithm). If neither is
    given, default key paths are used for every algorithm encountered.
    """
    resolved_paths: Dict[str, str] = dict(public_key_paths or {})
    if public_key_path is not None:
        resolved_paths.setdefault(DEFAULT_SCHEME.algorithm_id, public_key_path)

    public_key_cache: Dict[str, Any] = {}

    def _get_public_key(algorithm_id: str):
        if algorithm_id in public_key_cache:
            return public_key_cache[algorithm_id]
        scheme = SIGNATURE_SCHEMES.get(algorithm_id)
        if scheme is None:
            raise ValueError(f"Unknown signature algorithm in log: {algorithm_id!r}")
        path = resolved_paths.get(algorithm_id)
        key = load_public_key(path, scheme=scheme)
        public_key_cache[algorithm_id] = key
        return key

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
            sig_algorithm = row.get("sig_algorithm", "") or DEFAULT_SCHEME.algorithm_id
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

            # 2. Content integrity (including the claimed algorithm itself)
            recomputed_hash = compute_entry_hash(business_row, timestamp_utc, prev_hash, sig_algorithm)
            if recomputed_hash != entry_hash:
                return VerificationResult(
                    ok=False, rows_checked=rows_checked, first_failure_row=i,
                    failure_reason=(
                        f"entry_hash mismatch: recomputed={recomputed_hash[:16]}..., "
                        f"stored={entry_hash[:16]}.... "
                        "This indicates the content of this row (or its claimed signature "
                        "algorithm) was edited after it was written."
                    ),
                )

            # 3. Signature authenticity, dispatched to whichever scheme this
            # row claims to be signed under
            try:
                scheme = SIGNATURE_SCHEMES.get(sig_algorithm)
                if scheme is None:
                    raise ValueError(f"unknown signature algorithm {sig_algorithm!r}")
                public_key = _get_public_key(sig_algorithm)
                scheme.verify(public_key, bytes.fromhex(signature_hex), bytes.fromhex(entry_hash))
            except (InvalidSignature, ValueError) as e:
                return VerificationResult(
                    ok=False, rows_checked=rows_checked, first_failure_row=i,
                    failure_reason=(
                        f"signature invalid: {e}. This indicates the row's signature does not "
                        "match the provided public key -- either forged, corrupted, signed "
                        "with a different key, or claiming an unsupported algorithm."
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
    p_verify.add_argument("--pubkey", default=None, help="Path to the signer's public key (default scheme)")

    p_genkey = sub.add_parser("init-key", help="Generate a local signing keypair if one doesn't exist yet")
    p_genkey.add_argument("--private", default=None)
    p_genkey.add_argument("--public", default=None)

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
# - Post-quantum SignatureScheme implementation, once NIST actually
#   finalizes a standard (not before -- adopting an unstandardized algorithm
#   today would be a worse bet than waiting).
