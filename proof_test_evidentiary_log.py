"""
Proof test for failguard_evidentiary_log.py — run this, don't just trust it.

Tests:
  1. Normal append + verify -> should PASS
  2. Content tampering (edit a field after the fact) -> should FAIL, pinpointed
  3. Row deletion (remove a historical entry) -> should FAIL, pinpointed
  4. Row reordering -> should FAIL, pinpointed
  5. Signature forgery (different key) -> should FAIL, pinpointed
  6. Mining usability -> prove pandas can still read it normally
"""

import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "supervisor"))

from failguard_evidentiary_log import (
    EvidentiaryLogWriter,
    verify_log,
    load_or_create_keypair,
    ALL_FIELDNAMES,
)

TEST_DIR = "test_evidentiary_run"
LOG_PATH = os.path.join(TEST_DIR, "results.csv")
KEY_DIR = os.path.join(TEST_DIR, "keys")
PRIV_KEY = os.path.join(KEY_DIR, "key.pem")
PUB_KEY = os.path.join(KEY_DIR, "key.pub")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def reset():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)


def write_sample_log(n=5):
    writer = EvidentiaryLogWriter(LOG_PATH, private_key_path=PRIV_KEY, public_key_path=PUB_KEY,
                                   async_write=False)
    for i in range(n):
        writer.append({
            "idx": i,
            "prompt": f"Schedule a routine checkup for patient {i}",
            "label": 0,
            "label_str": "benign",
            "status": "SAFE",
            "correct": True,
            "similarity": 0.12,
            "benign_sim": 0.91,
            "risky_votes": 0,
            "benign_votes": 2,
            "confidence": "high",
            "matched_mode": "",
            "matched_category": "",
            "triggered_on": "prompt",
            "reranked": False,
            "reranker_verdict": "",
            "reranker_reason": "",
            "reranker_q6": "",
            "grok_invoked": False,
            "grok_response": "",
            "explanation": "Low risky similarity, strong benign anchor match.",
        })
    return writer


def read_rows():
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def run():
    results = []

    # --- Test 1: normal append + verify ---
    reset()
    write_sample_log(5)
    r = verify_log(LOG_PATH, PUB_KEY)
    ok = r.ok and r.rows_checked == 5
    results.append(("1. Clean log verifies PASS", ok, str(r)))

    # --- Test 2: content tampering ---
    reset()
    write_sample_log(5)
    rows = read_rows()
    rows[2]["status"] = "INTERVENE"  # flip a verdict after the fact
    write_rows(rows)
    r = verify_log(LOG_PATH, PUB_KEY)
    ok = (not r.ok) and r.first_failure_row == 3
    results.append(("2. Edited field detected at correct row", ok, str(r)))

    # --- Test 3: row deletion ---
    reset()
    write_sample_log(5)
    rows = read_rows()
    del rows[2]  # remove a historical entry
    write_rows(rows)
    r = verify_log(LOG_PATH, PUB_KEY)
    ok = (not r.ok) and r.first_failure_row == 3
    results.append(("3. Deleted row detected (chain break)", ok, str(r)))

    # --- Test 4: row reordering ---
    reset()
    write_sample_log(5)
    rows = read_rows()
    rows[2], rows[3] = rows[3], rows[2]
    write_rows(rows)
    r = verify_log(LOG_PATH, PUB_KEY)
    ok = (not r.ok) and r.first_failure_row == 3
    results.append(("4. Reordered rows detected (chain break)", ok, str(r)))

    # --- Test 5: signature forgery (wrong key signs a tampered entry_hash) ---
    reset()
    write_sample_log(5)
    rows = read_rows()
    # Recompute a fake hash/signature using a DIFFERENT key, simulating an
    # attacker who edits content and tries to re-sign it without the real key.
    from failguard_evidentiary_log import compute_entry_hash, BUSINESS_FIELDNAMES
    forged_priv = load_or_create_keypair(
        os.path.join(TEST_DIR, "attacker_key.pem"),
        os.path.join(TEST_DIR, "attacker_key.pub"),
    )
    target = rows[2]
    business = {k: target[k] for k in BUSINESS_FIELDNAMES}
    business["status"] = "INTERVENE"
    new_hash = compute_entry_hash(business, target["timestamp_utc"], target["prev_hash"],
                                   target.get("sig_algorithm", "ed25519"))
    new_sig = forged_priv.sign(bytes.fromhex(new_hash)).hex()
    rows[2]["status"] = "INTERVENE"
    rows[2]["entry_hash"] = new_hash
    rows[2]["signature"] = new_sig
    # also have to patch the next row's prev_hash to keep chain superficially consistent
    rows[3]["prev_hash"] = new_hash
    write_rows(rows)
    r = verify_log(LOG_PATH, PUB_KEY)  # verifying against the REAL public key
    ok = (not r.ok) and r.first_failure_row == 3
    results.append(("5. Forged entry (wrong signing key) detected", ok, str(r)))

    # --- Test 6: still minable with pandas ---
    reset()
    write_sample_log(5)
    try:
        import pandas as pd
        df = pd.read_csv(LOG_PATH)
        ok = len(df) == 5 and "status" in df.columns and "similarity" in df.columns
        detail = f"pandas loaded {len(df)} rows, columns include business fields normally"
    except Exception as e:
        ok = False
        detail = str(e)
    results.append(("6. Log remains directly minable (pandas)", ok, detail))

    # --- Test 7: async mode doesn't block the caller ---
    reset()
    import time
    writer = EvidentiaryLogWriter(LOG_PATH, private_key_path=PRIV_KEY, public_key_path=PUB_KEY,
                                   async_write=True)
    t0 = time.perf_counter()
    writer.append({"idx": 0, "prompt": "p0", "status": "INTERVENE"})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    # append() should return in well under a millisecond -- it's just a
    # queue.put(), no signing, no disk I/O. Generous threshold (5ms) to
    # avoid CI flakiness while still proving it's not doing fsync inline.
    ok = elapsed_ms < 5.0
    results.append(("7. append() returns near-instantly in async mode", ok,
                     f"{elapsed_ms:.3f}ms (threshold: 5ms)"))

    # --- Test 8: flush() actually waits for the write to complete ---
    writer.flush()
    r = verify_log(LOG_PATH, PUB_KEY)
    ok = r.ok and r.rows_checked == 1
    results.append(("8. flush() guarantees the write landed before returning", ok, str(r)))

    # --- Test 9: concurrent rapid-fire appends still produce a valid, ---
    # --- correctly-ordered chain (the actual race-condition claim) -----
    reset()
    writer = EvidentiaryLogWriter(LOG_PATH, private_key_path=PRIV_KEY, public_key_path=PUB_KEY,
                                   async_write=True)
    import threading as _threading
    N_THREADS = 8
    N_PER_THREAD = 15

    def hammer(thread_id):
        for i in range(N_PER_THREAD):
            writer.append({"idx": f"{thread_id}-{i}", "prompt": f"thread {thread_id} item {i}",
                            "status": "SAFE" if i % 2 == 0 else "INTERVENE"})

    threads = [_threading.Thread(target=hammer, args=(t,)) for t in range(N_THREADS)]
    for t in threads: t.start()
    for t in threads: t.join()
    writer.flush()

    r = verify_log(LOG_PATH, PUB_KEY)
    expected_count = N_THREADS * N_PER_THREAD
    ok = r.ok and r.rows_checked == expected_count
    failed = writer.get_failed_writes()
    detail = f"{r}, {N_THREADS} threads x {N_PER_THREAD} writes = {expected_count} expected, " \
             f"{len(failed)} failed writes"
    results.append(("9. Concurrent writes from 8 threads -> valid ordered chain, zero corruption",
                     ok and len(failed) == 0, detail))

    # --- Test 10: tampering with the RECORDED ALGORITHM itself is caught ---
    # (proves sig_algorithm is actually part of the hashed payload, not just
    # an unprotected metadata column)
    reset()
    write_sample_log(5)
    rows = read_rows()
    rows[2]["sig_algorithm"] = "some_future_pq_scheme"  # tamper with the claimed algorithm only
    write_rows(rows)
    r = verify_log(LOG_PATH, PUB_KEY)
    ok = (not r.ok) and r.first_failure_row == 3
    results.append(("10. Tampering with sig_algorithm field alone is detected", ok, str(r)))

    # --- Test 11: a genuinely different second scheme coexists in the ---
    # --- same chain (the actual proof the abstraction/seam works) -------
    reset()
    from failguard_evidentiary_log import SignatureScheme, SIGNATURE_SCHEMES
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization as _ser

    class RSATestScheme(SignatureScheme):
        """A second, real, independent algorithm -- not Ed25519 -- used here
        purely to prove the registry/dispatch mechanism genuinely supports
        more than one scheme. Stand-in for a future real PQC scheme."""
        algorithm_id = "rsa_test"

        def generate_private_key(self):
            return rsa.generate_private_key(public_exponent=65537, key_size=2048)

        def serialize_private_key(self, key) -> bytes:
            return key.private_bytes(_ser.Encoding.PEM, _ser.PrivateFormat.PKCS8, _ser.NoEncryption())

        def deserialize_private_key(self, data: bytes):
            return _ser.load_pem_private_key(data, password=None)

        def public_key_from_private(self, private_key):
            return private_key.public_key()

        def serialize_public_key(self, key) -> bytes:
            return key.public_bytes(_ser.Encoding.PEM, _ser.PublicFormat.SubjectPublicKeyInfo)

        def deserialize_public_key(self, data: bytes):
            return _ser.load_pem_public_key(data)

        def sign(self, private_key, message: bytes) -> bytes:
            return private_key.sign(message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                     salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())

        def verify(self, public_key, signature: bytes, message: bytes) -> None:
            public_key.verify(signature, message,
                               padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                           salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())

    SIGNATURE_SCHEMES["rsa_test"] = RSATestScheme()

    # First two entries signed under Ed25519 (the default)
    w1 = EvidentiaryLogWriter(LOG_PATH, private_key_path=PRIV_KEY, public_key_path=PUB_KEY,
                               async_write=False)
    w1.append({"idx": 0, "prompt": "ed25519 entry A", "status": "SAFE"})
    w1.append({"idx": 1, "prompt": "ed25519 entry B", "status": "SAFE"})

    # Next two entries signed under the new RSA scheme -- same file, continuing the SAME chain
    rsa_priv = f"{TEST_DIR}/keys/rsa_key.pem"
    rsa_pub = f"{TEST_DIR}/keys/rsa_key.pub"
    w2 = EvidentiaryLogWriter(LOG_PATH, private_key_path=rsa_priv, public_key_path=rsa_pub,
                               async_write=False, scheme=RSATestScheme())
    w2.append({"idx": 2, "prompt": "rsa entry C", "status": "INTERVENE"})
    w2.append({"idx": 3, "prompt": "rsa entry D", "status": "SAFE"})

    # Verify the WHOLE mixed-algorithm chain in one pass, supplying both public keys
    r = verify_log(LOG_PATH, public_key_paths={"ed25519": PUB_KEY, "rsa_test": rsa_pub})
    ok = r.ok and r.rows_checked == 4
    results.append(("11. Mixed-algorithm chain (Ed25519 + a second real scheme) verifies as one chain",
                     ok, str(r)))

    # --- Report ---
    print("\n" + "=" * 70)
    print("EVIDENTIARY LOG -- PROOF TEST RESULTS")
    print("=" * 70)
    all_ok = True
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"\n[{status}] {name}")
        print(f"        {detail}")
        all_ok = all_ok and ok

    print("\n" + "=" * 70)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 70)

    shutil.rmtree(TEST_DIR)
    return all_ok


if __name__ == "__main__":
    success = run()
    raise SystemExit(0 if success else 1)
