"""
FailGuard Supervisor - The Prevention Engine
Monitors agent trajectories in real time using taxonomy + multidimensional mapping.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.taxonomy import FailGuardTaxonomy
from src.core.mapper import MultidimensionalMapper
from typing import Dict, Any, Optional

class FailGuardSupervisor:
    def __init__(self, drift_threshold: float = 0.45):
        print("🚀 Initializing FailGuard Supervisor...")
        self.taxonomy = FailGuardTaxonomy()
        self.mapper = MultidimensionalMapper()
        self.drift_threshold = drift_threshold
        self.trajectory_history = []
        print("✅ FailGuard Supervisor ready for real-time prevention")

    def evaluate_step(self, proposed_action: str, context: str = "") -> Dict[str, Any]:
        """
        Evaluate the next proposed action / trajectory step.
        Returns a decision: OK, WARN, or INTERVENE.
        """
        # Build current trajectory text
        trajectory_text = f"CONTEXT: {context}\nPROPOSED ACTION: {proposed_action}"
        
        # Detect drift in high-dimensional space
        drift_result = self.mapper.detect_drift(trajectory_text, self.drift_threshold)
        
        # Get enterprise impact and legal notes
        impact = self.taxonomy.get_enterprise_impact(drift_result.get("closest_failure", ""))
        legal = self.taxonomy.get_legal_notes(drift_result.get("closest_failure", ""))
        
        decision = {
            "status": "INTERVENE" if drift_result["is_drifting"] else "OK",
            "closest_failure": drift_result["closest_failure"],
            "category": drift_result["category"],
            "distance": drift_result["distance"],
            "enterprise_impact": impact,
            "legal_notes": legal,
            "recommendation": (
                "⚠️  HIGH RISK - Intervene immediately" if drift_result["is_drifting"]
                else "✅ Safe to proceed"
            )
        }
        
        # Log to history
        self.trajectory_history.append({
            "action": proposed_action,
            "decision": decision
        })
        
        return decision

    def get_history(self) -> list:
        """Return full trajectory decision history."""
        return self.trajectory_history


# Quick demo / test
if __name__ == "__main__":
    supervisor = FailGuardSupervisor()
    
    print("\n🧪 Testing FailGuard Supervisor...\n")
    
    test_cases = [
        "The agent plans to call a tool that deletes user data without any confirmation step.",
        "The agent confidently cites a regulation that was repealed last year.",
        "The agent is about to send a polite email to the customer."
    ]
    
    for i, action in enumerate(test_cases, 1):
        print(f"Test {i}: {action[:80]}...")
        result = supervisor.evaluate_step(action)
        print(f"   → {result['recommendation']}")
        print(f"   Closest failure: {result['closest_failure']}\n")
    
    print("✅ Supervisor test complete!")