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
    writer = EvidentiaryLogWriter(LOG_PATH, private_key_path=PRIV_KEY, public_key_path=PUB_KEY)
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
    new_hash = compute_entry_hash(business, target["timestamp_utc"], target["prev_hash"])
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
