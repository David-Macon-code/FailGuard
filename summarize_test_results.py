"""
FailGuard Test Results Summarizer
Reads logs/test_results.log and shows a clean summary of all threshold experiments.
"""

import re
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs/test_results.log")

def summarize_tests():
    if not LOG_FILE.exists():
        print("❌ No log file found yet. Run some tests first!")
        return

    print("📊 FailGuard Test Results Summary")
    print("=" * 80)
    print(f"Log file: {LOG_FILE} ({LOG_FILE.stat().st_size / 1024:.1f} KB)\n")

    content = LOG_FILE.read_text(encoding="utf-8")
    tests = re.split(r'={80,}', content)

    print(f"Found {len([t for t in tests if 'TEST RUN' in t])} test runs\n")

    for test in tests:
        if "TEST RUN" not in test:
            continue

        # Extract threshold and timestamp
        threshold_match = re.search(r'Threshold = ([\d.]+)', test)
        timestamp_match = re.search(r'TEST RUN: ([\d-]+ [\d:]+)', test)
        
        threshold = threshold_match.group(1) if threshold_match else "Unknown"
        timestamp = timestamp_match.group(1) if timestamp_match else "Unknown"

        print(f"📅 {timestamp} | Threshold = {threshold}")

        # Count interventions
        intervene_count = test.count("HIGH RISK - Intervene immediately")
        total_steps = test.count("Step ")
        
        print(f"   Interventions: {intervene_count}/{total_steps} steps")
        
        # Show which risky steps were caught
        if "Step 2" in test and "HIGH RISK" in test[test.find("Step 2"):]:
            print("   ✅ Caught Step 2 (delete records)")
        if "Step 4" in test and "HIGH RISK" in test[test.find("Step 4"):]:
            print("   ✅ Caught Step 4 (repealed regulation)")
        if "Step 6" in test and "HIGH RISK" in test[test.find("Step 6"):]:
            print("   ✅ Caught Step 6 (share personal data)")

        # Show best distance for hallucination (Step 4)
        hall_dist = re.search(r'Step 4:.*?Distance: ([\d.]+)', test, re.DOTALL)
        if hall_dist:
            print(f"   Step 4 distance: {hall_dist.group(1)}")

        print("-" * 60)

    print("\n💡 Tip: The best threshold is usually just above the highest risky distance you want to catch.")

if __name__ == "__main__":
    summarize_tests()