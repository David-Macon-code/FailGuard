"""
FailGuard Taxonomy Loader
Loads the hybrid taxonomy from config/taxonomy_config.yaml
Built from David's two AI Failure Taxonomy courses + supporting research
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

class FailGuardTaxonomy:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent.parent / "config" / "taxonomy_config.yaml"
        self.taxonomy = self._load_taxonomy()
        print(f"✅ Loaded FailGuard Taxonomy v{self.taxonomy.get('version', 'unknown')}")

    def _load_taxonomy(self) -> Dict:
        """Load and parse the YAML taxonomy file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Taxonomy config not found at {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_categories(self) -> List[str]:
        """Return list of all top-level categories."""
        return [cat["name"] for cat in self.taxonomy.get("categories", [])]

    def get_sub_modes(self, category_name: str) -> List[Dict]:
        """Get all sub-modes for a specific category."""
        for cat in self.taxonomy.get("categories", []):
            if cat["name"].lower() == category_name.lower():
                return cat.get("sub_modes", [])
        return []

    def search(self, keyword: str) -> List[Dict]:
        """Simple keyword search across all sub-modes."""
        results = []
        keyword = keyword.lower()
        for cat in self.taxonomy.get("categories", []):
            for mode in cat.get("sub_modes", []):
                if (keyword in mode.get("name", "").lower() or
                    keyword in mode.get("description", "").lower() or
                    keyword in mode.get("enterprise_impact", "").lower()):
                    results.append({
                        "category": cat["name"],
                        "mode": mode
                    })
        return results

    def get_enterprise_impact(self, mode_name: str) -> Optional[str]:
        """Quick lookup for business impact of a specific failure mode."""
        for cat in self.taxonomy.get("categories", []):
            for mode in cat.get("sub_modes", []):
                if mode.get("name", "").lower() == mode_name.lower():
                    return mode.get("enterprise_impact")
        return None

    def get_legal_notes(self, mode_name: str) -> Optional[str]:
        """Quick lookup for legal/compliance notes."""
        for cat in self.taxonomy.get("categories", []):
            for mode in cat.get("sub_modes", []):
                if mode.get("name", "").lower() == mode_name.lower():
                    return mode.get("legal_notes")
        return None

    def list_all_modes(self) -> List[Dict]:
        """Return every failure mode with its category for easy iteration."""
        all_modes = []
        for cat in self.taxonomy.get("categories", []):
            for mode in cat.get("sub_modes", []):
                all_modes.append({
                    "category": cat["name"],
                    "name": mode.get("name"),
                    "description": mode.get("description"),
                    "enterprise_impact": mode.get("enterprise_impact"),
                    "legal_notes": mode.get("legal_notes")
                })
        return all_modes


# Quick test when run directly
if __name__ == "__main__":
    taxonomy = FailGuardTaxonomy()
    print(f"Loaded {len(taxonomy.list_all_modes())} failure modes")
    print("\nExample categories:", taxonomy.get_categories()[:6])
    
    # Demo search
    results = taxonomy.search("hallucination")
    if results:
        print(f"\nFound {len(results)} matches for 'hallucination'")