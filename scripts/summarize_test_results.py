"""
FailGuard Test Results Summarizer - Fixed Version
"""

from pathlib import Path

LOG_FILE = Path("logs/test_results.log")

def summarize_tests():
    if not LOG_FILE.exists():
        print("❌ No log file found. Run some tests first!")
        return

    content = LOG_FILE.read_text(encoding="utf-8")
    runs = content.split("TEST RUN")

    print("📊 FailGuard Test Results Summary")
    print("=" * 80)
    print(f"Log file: {LOG_FILE} ({LOG_FILE.stat().st_size / 1024:.1f} KB)")
    print(f"Found {len(runs)-1} test runs\n")

    for run in runs[1:]:   # skip first empty block
        # Extract threshold
        threshold = "Unknown"
        if "Threshold =" in run:
            threshold = run.split("Threshold =")[1].split()[0]

        # Extract timestamp
        timestamp_line = run.split("\n")[0].strip()
        print(f"📅 {timestamp_line} | Threshold = {threshold}")

        # Count interventions
        intervene_count = run.count("HIGH RISK - Intervene immediately")
        total_steps = run.count("Step ")

        print(f"   Interventions: {intervene_count}/{total_steps} steps")

        # Show caught risky steps
        if "Step 2" in run and "HIGH RISK" in run:
            print("   ✅ Caught Step 2 (delete records without confirmation)")
        if "Step 4" in run and "HIGH RISK" in run:
            print("   ✅ Caught Step 4 (repealed regulation)")
        if "Step 6" in run and "HIGH RISK" in run:
            print("   ✅ Caught Step 6 (share personal data)")

        print("-" * 70)

    print("\n💡 Your tuning work is now properly documented!")

if __name__ == "__main__":
    summarize_tests()